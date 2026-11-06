"""Build the frozen-source Colab notebook and user-facing package."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / "outputs" / "CFRI_holdout_v1"
NOTEBOOK = DEST / "CFRI_holdout_v1_Colab.ipynb"
SOURCES = (
    "PROTOCOL.md", "README.md", "requirements-colab.txt", "cfri_core.py",
    "upstream/cfri_core.py",
    "confirm_saits.py", "v2_price.py", "v2_peer.py", "holdout_data.py",
    "holdout_run.py", "holdout_saits.py", "holdout_analyze.py",
    "test_holdout_core.py", "test_holdout_saits.py", "test_holdout_analyze.py",
    "build_notebook.py",
)


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def stable_zip(files: list[Path], root: Path) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            z.writestr(entry, path.read_bytes())
    return data.getvalue()


def main() -> None:
    missing = [name for name in SOURCES if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sources: {missing}")
    DEST.mkdir(parents=True, exist_ok=True)
    code = DEST / "code"
    code.mkdir(exist_ok=True)
    for name in SOURCES:
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, code / name)
    manifest = {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in SOURCES}
    (code / "SOURCE_SHA256.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    sources = [code / name for name in sorted((*SOURCES, "SOURCE_SHA256.json"))]
    packed = stable_zip(sources, code)
    digest = hashlib.sha256(packed).hexdigest()
    encoded = base64.b64encode(packed).decode()

    intro = """# CFRI simple holdout v1 — yeni varlıklarda kilitli sınama

GitHub yüklemesini koşu tamamlandıktan sonra yapabilirsiniz. İlk **Run all**, gömülü kodun SHA-256 değerini ve UTC zamanını Google Drive'a kaydeder; veri indirme bu yerel dondurma kaydından sonra başlar. Bu kayıt dışarıdan doğrulanmış bir ön kayıt zaman damgası değildir. Protokol ve kapsam sınırları `PROTOCOL.md` içindedir.

Bu deney daha önce kullanılmamış 16 ETF'yi hedef alır; **2026 Ocak–Eylül dönemi aynı kaldığından yeni zaman dönemi doğrulaması değildir**. Tek tek alanlar ve tam satır kesintileri ayrı raporlanır. CFRI/SAITS aynı maskeleri kullanır; tam satır karşılaştırmasında ikisine de aynı peer köprüsü verilir. Özel Yahoo ham dosyaları sonuç ZIP'ine eklenmez.

Colab'da **Runtime > Change runtime type > GPU**, tercihen A100, seçin ve **Run all** deyin. Google Drive'a izin verin. Sonuçlar `MyDrive/ITRI/CFRI_holdout_v1_local` klasörüne koşul bazında kaydedilir. Oturum koparsa aynı notebook'ta yeniden **Run all** yapın; SHA ile doğrulanan koşullar atlanır, yalnız yarım kalan koşul tekrar hesaplanır. Son ZIP otomatik indirilir.
"""
    deps = """import subprocess, sys
requirements = """ + repr([line.strip() for line in (HERE / "requirements-colab.txt").read_text().splitlines() if line.strip()]) + """
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
print("Pinned Python packages installed; PyTorch comes from the selected Colab runtime.")
"""
    setup = f'''import base64, hashlib, io, json, os, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
run_root = Path("/content/drive/MyDrive/ITRI/CFRI_holdout_v1_local")
run_root.mkdir(parents=True, exist_ok=True)
code_dir = Path("/content/cfri_holdout_v1_code")
code_dir.mkdir(parents=True, exist_ok=True)
packed = base64.b64decode({encoded!r})
embedded_bundle_sha256 = {digest!r}
assert hashlib.sha256(packed).hexdigest() == embedded_bundle_sha256
with zipfile.ZipFile(io.BytesIO(packed)) as z:
    if z.testzip() is not None:
        raise RuntimeError("Embedded code ZIP CRC failed")
    z.extractall(code_dir)
source_hashes = json.loads((code_dir / "SOURCE_SHA256.json").read_text())
for name, expected in source_hashes.items():
    if hashlib.sha256((code_dir / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Source hash mismatch: {{name}}")
print("Drive results:", run_root)
print("Embedded frozen source SHA-256:", embedded_bundle_sha256)
'''
    freeze = '''source_path = run_root / "frozen_source_bundle.zip"
if source_path.exists():
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != embedded_bundle_sha256:
        raise RuntimeError("Drive contains a different frozen source bundle; use the original notebook or a new root")
else:
    temp = source_path.with_name(source_path.name + ".tmp")
    temp.write_bytes(packed)
    os.replace(temp, source_path)
receipt_path = run_root / "freeze_receipt.json"
if receipt_path.exists():
    freeze_receipt = json.loads(receipt_path.read_text())
    if freeze_receipt.get("embedded_bundle_sha256") != embedded_bundle_sha256:
        raise RuntimeError("Freeze receipt changed or does not match this notebook")
else:
    freeze_receipt = {"embedded_bundle_sha256": embedded_bundle_sha256,
                      "freeze_utc": datetime.now(timezone.utc).isoformat()}
    temp = receipt_path.with_name(receipt_path.name + ".tmp")
    temp.write_text(json.dumps(freeze_receipt, sort_keys=True, indent=2) + "\\n")
    os.replace(temp, receipt_path)
print("Freeze receipt:", freeze_receipt)
'''
    runner = '''import subprocess
from collections import deque

def run_stage(name, script, args):
    recent = deque(maxlen=60)
    command = [sys.executable, "-u", str(code_dir / script), "--root", str(run_root), *args]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=str(code_dir), text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            recent.append(line)
        status = process.wait()
    if status:
        detail = "".join(recent)
        (run_root / f"{name}_last_error.log").write_text(detail)
        raise RuntimeError(f"{name} stopped (exit {status}). Last output:\\n{detail}")
'''
    tests = '''test_files = sorted(p.stem for p in code_dir.glob("test_holdout_*.py"))
subprocess.check_call([sys.executable, "-m", "unittest", "-q", *test_files], cwd=str(code_dir))
print("Synthetic tests passed:", test_files)
'''
    data = '''run_stage("data", "holdout_run.py", ["--stage", "data"])
'''
    cpu = '''run_stage("cpu", "holdout_run.py", ["--stage", "cpu"])
'''
    neural = '''saits_workers = None  # Automatic: up to 4 on an A100; 1 on smaller GPUs.
saits_args = [] if saits_workers is None else ["--workers", str(saits_workers)]
run_stage("saits", "holdout_saits.py", saits_args)
'''
    finish = '''from google.colab import files
run_stage("analyze", "holdout_analyze.py", [])
result_zip = run_root / "CFRI_holdout_v1_results.zip"
if not result_zip.is_file():
    raise FileNotFoundError(result_zip)
print("Completed ZIP:", result_zip)
files.download(str(result_zip))
'''
    notebook = {
        "cells": [cell("markdown", intro), cell("code", deps), cell("code", setup),
                  cell("code", freeze), cell("code", runner), cell("code", tests),
                  cell("code", data), cell("code", cpu), cell("code", neural), cell("code", finish)],
        "metadata": {"colab": {"name": NOTEBOOK.name, "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    for name in ("README.md", "PROTOCOL.md", "requirements-colab.txt", "SOURCE_SHA256.json"):
        shutil.copyfile(code / name, DEST / name)
    package = DEST / "CFRI_holdout_v1_package.zip"
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p != package)
    package.write_bytes(stable_zip(files, DEST))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:
            raise RuntimeError("Package ZIP CRC failed")
    print("Notebook:", NOTEBOOK, NOTEBOOK.stat().st_size)
    print("Package:", package, package.stat().st_size)
    print("Embedded bundle SHA-256:", digest)


if __name__ == "__main__":
    main()
