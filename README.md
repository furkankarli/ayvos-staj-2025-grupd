# AYVOS Staj 2025 – Grup D

Bu repository, 4 kişilik staj ekibimizin ortak pratik çalışmalarını barındırır.
Her klasör ayrı bir ödevi temsil eder.
**Kalite Koruması:** Pre-commit + GitHub Actions PEP8 testi + En az 2 onaylı PR.

---

## Proje Kurulumu (Sadece Bir Kez)

1. **Depoyu Klonla**

   ```bash
   git clone https://github.com/furkankarli/ayvos-staj-2025-grupd.git
   cd ayvos-staj-2025-grupd
   ```

2. **Gereksinimleri Yükle**

   ```bash
   python -m pip install --upgrade pip
   pip install pre-commit
   pre-commit install
   ```

---

## Günlük Çalışma Akışı

| Adım                | Komut / Açıklama                                                           |
| ------------------- | -------------------------------------------------------------------------- |
| **1. Branch Aç**    | `git checkout -b ad-soyad/ozellik`                                         |
| **2. Kod Yaz**      | `src/` veya `notebooks/` altına yeni dosya ekle                            |
| **3. Lint & Test**  | `pre-commit run --all-files` (isteğe bağlı)                                |
| **4. Commit**       | `git add .<br>git commit -m "feat: kısa açıklama"`                 |
| **5. Push & PR**    | `git push -u origin ad-soyad/ozellik`GitHub’da Pull Request aç |
| **6. Onay & Merge** | En az **2 onay** + **PEP8 Check** yeşil → `main`’e merge                   |

---

## Klasör Yapısı

```
ayvos-staj-2025-grupd/
├── .github/
│   └── workflows/
│       └── pep8.yml           # CI için PEP8 testleri
├── .pre-commit-config.yaml    # Yerel lint/format ayarları
├── README.md                  # Proje açıklama dosyası
├── requirements.txt           # Ortak Python paketleri
└── 01-semantic-vs-instance/   # Ödev klasörü (örnek)
    ├── README.md
    ├── requirements.txt
    ├── src/
    └── assets/
```

---

## Lint & Format Araçları

| Araç       | İşlev                                   |
| ---------- | --------------------------------------- |
| **Black**  | Kodu otomatik olarak 88 sütuna uyarlama |
| **isort**  | Import’ları alfabetik sıraya koyma      |
| **flake8** | PEP 8 ihlallerini tespit etme           |

### Elle Test

* Tüm dosyaları kontrol et:

  ```bash
  pre-commit run --all-files
  ```
* Belirli bir dosya:

  ```bash
  pre-commit run --files path/to/file.py
  ```

---

## Sık Karşılaşılan Hatalar & Çözümler

| Hata                            | Çözüm                                                         |
| ------------------------------- | ------------------------------------------------------------- |
| `E902 FileNotFoundError`        | Kök dizinde .py dosyası yok → Yeni bir Python dosyası ekleyin |
| Black format sonrası değişiklik | `git add .` ile formatlanan dosyaları tekrar ekleyin          |
| Flake8 hatası                   | Kodunuzu PEP 8’a uygun hale getirin veya `# noqa` kullanın    |

---

