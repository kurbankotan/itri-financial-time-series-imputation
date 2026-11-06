# CFRI simple holdout v1

Bu paket, sadeleştirilmiş CFRI'yi önceki 34 hedef ETF arasında bulunmayan, önceden belirlenmiş 16 ETF'de sınar. Önceki 34 ülke ETF'si sabit karşılaştırma havuzudur. Bu çalışma **yeni varlık sınamasıdır**; 2026 Ocak–Eylül dönemi daha önce incelendiğinden yeni zaman dönemi doğrulaması değildir.

## Çalıştırma sırası

1. `CFRI_holdout_v1_Colab.ipynb` dosyasını Colab'da açın, GPU çalışma zamanı seçin (tercihen A100) ve **Run all** deyin. Google Drive iznini verin. Notebook veri indirmeden önce gömülü kaynak paketini, SHA-256 değerini ve UTC zamanını `MyDrive/ITRI/CFRI_holdout_v1_local` klasörüne kaydeder. GitHub commit'i istenmez.
2. Notebook bağımlılıkları kurar, sentetik testleri çalıştırır, veriyi indirip kontrol eder, CPU ve SAITS koşullarını yürütür, analizi yapar ve `CFRI_holdout_v1_results.zip` dosyasını otomatik indirir.
3. Koşu bittikten sonra bu paketin protokolünü ve kodunu GitHub'a yükleyebilirsiniz. Özel ham Yahoo fiyatlarını ve Drive'daki `raw_do_not_publish` klasörünü yüklemeyin. Sonuçlardan sonraki GitHub commit'i, koşudan önce kamuya açık bir ön kayıt yapıldığını kanıtlamaz; rapor sonuçları **yerel olarak dondurulmuş yeni varlık sınaması** diye etiketler.

Drive klasöründe özel Yahoo ham verileri ve koşul kayıtları tutulur. Colab bağlantısı kesilirse aynı notebook'ta yeniden **Run all** deyin. SHA ile doğrulanan tamamlanmış koşullar atlanır; yarım kalan SAITS koşulu yeniden eğitilir. Sonuç ZIP'ine ham fiyatlar eklenmez.

Yahoo indirmesi hata verirse yöntemler başlamadan veri adımı durur. Sağlayıcı düzeldikten sonra **Run all** tekrar çalıştırılabilir; bu koşunun Drive klasöründe başarıyla inmiş dosyalar hash denetiminden sonra kullanılır.

Sabit varlık listesi, kalite kuralları, yöntemler, karşılaştırmalar ve yorum sınırları `PROTOCOL.md` dosyasındadır. Kaynak hash'leri `SOURCE_SHA256.json` içindedir. İlk çalıştırmada oluşan yerel dondurma kaydından sonraki kod değişiklikleri ayrıca sürümlenmeli ve açıklanmalıdır.
