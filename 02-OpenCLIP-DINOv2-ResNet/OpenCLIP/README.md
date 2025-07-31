# OpenCLIP Araç Sınıflandırma Projesi

Bu proje, OpenCLIP (Contrastive Language-Image Pre-training) modelini kullanarak araç sınıflandırma işlemlerini gerçekleştiren kapsamlı bir çalışmadır. Proje, zero-shot learning, prompt engineering, model karşılaştırması ve fine-tuning gibi farklı yaklaşımları içermektedir.

## 📁 Proje Yapısı

```
OpenCLIP/
├── prompt-test.py                    # Prompt engineering testleri
├── models-tests.py                   # Model karşılaştırma testleri
├── zero-shot/                        # Zero-shot sınıflandırma
│   ├── zero_shot_vehicle.py         # Ana zero-shot sınıflandırıcı
│   ├── zero_shot.py                 # Basit zero-shot örneği
│   ├── zeroshot_results.json        # Zero-shot sonuçları
│   └── confusion_matrix.png         # Karışıklık matrisi
├── train/                           # Model eğitimi
│   ├── predict_similarity.py        # Eğitilmiş model tahminleri
│   ├── evulate_model.py             # Model değerlendirme
│   ├── split_cfv_dataset.py         # Veri seti bölme
│   └── README.md                    # Eğitim konfigürasyonu
├── test_images/                     # Test görselleri
├── results/                         # Tahmin sonuçları
├── logs/                           # Eğitim logları
├── ensemble_results.txt             # Ensemble sonuçları
├── prompt_template_results.csv      # Prompt template sonuçları
├── model_comparison_results.csv    # Model karşılaştırma sonuçları
├── prompt_engineering_results.png   # Prompt engineering grafikleri
└── model_comparison.png            # Model karşılaştırma grafikleri
```

## 🚀 Ana Bileşenler

### 1. Prompt Engineering (`prompt-test.py`)

Bu modül, farklı prompt şablonlarının model performansına etkisini test eder.

**Özellikler:**
- 12 farklı prompt template'i test eder
- Ensemble prompt yaklaşımı
- Çok dilli prompt desteği (Türkçe, Almanca, Fransızca)
- Görselleştirme ve sonuç analizi

**Test Edilen Prompt Template'leri:**
- `basic`: "a photo of a {class}"
- `detailed`: "a high quality photo of a {class}"
- `contextual`: "a photo of a {class} on a road"
- `professional`: "a professional photograph of a {class}"
- `descriptive`: "a clear photo of a {class} car"
- `artistic`: "an artistic photo of a {class}"
- `realistic`: "a realistic photo of a {class} vehicle"
- `studio`: "a studio photo of a {class}"
- `outdoor`: "an outdoor photo of a {class}"
- `side_view`: "a side view photo of a {class}"
- `front_view`: "a front view photo of a {class}"
- `angle_view`: "an angled view photo of a {class}"

**Sonuçlar:**
- En iyi performans: `front_view` template (%21.93 güven)
- Ensemble yaklaşımı tek prompt'tan %0.0003 daha iyi
- Detaylı sonuçlar: `prompt_template_results.csv`

### 2. Model Karşılaştırması (`models-tests.py`)

Farklı OpenCLIP modellerinin performansını karşılaştırır.

**Test Edilen Modeller:**
- ViT-B-32 (OpenAI)
- ViT-B-16 (OpenAI)
- ViT-L-14 (OpenAI)
- ViT-B-32 (LAION-2B)
- ViT-L-14 (LAION-2B)
- RN50 (OpenAI)
- RN101 (OpenAI)
- ConvNeXT-Base (LAION-2B)

**Karşılaştırma Metrikleri:**
- Yükleme süresi
- Inference süresi
- Model parametre sayısı
- Tahmin güveni
- Doğruluk oranı

**Sonuçlar:**
- En hızlı: ViT-B-32 (10.94ms)
- En doğru: ViT-L-14 (100% güven)
- En büyük model: ViT-L-14 (427.6M parametre)
- Detaylı sonuçlar: `model_comparison_results.csv`

### 3. Zero-Shot Sınıflandırma (`zero-shot/`)

Eğitim gerektirmeden araç sınıflandırma yapar.

**Desteklenen Araç Sınıfları (44 sınıf):**
- Alfa Romeo Giulia
- Audi A4, A6
- BMW 3 Series, X3
- Citroen C3, C4 Grand Picasso
- Dacia Logan, Spring
- Fiat Bravo
- Ford Fiesta, Focus, Fusion, Mondeo, Transit
- Honda Civic
- Hyundai i30
- Kia Sportage
- Maserati Levante
- Mazda 2
- Mini Countryman
- Mitsubishi L200
- Opel Astra, Corsa, Meriva
- Peugeot 208, 3008
- Renault Captur
- Seat Ibiza, Leon
- Skoda Fabia, Octavia, Superb
- Smart Forfour, Fortwo
- Suzuki SX4 S-Cross, Vitara
- Tesla S
- Toyota C-HR, Corolla, Yaris
- Volkswagen Golf, Passat, Polo

**Performans Sonuçları:**
- Genel doğruluk: %35.8
- En iyi sınıflar: Ford Transit, Kia Sportage, Maserati Levante, Mini Countryman, Mitsubishi L200, Smart Fortwo, Tesla S, Toyota Yaris (%100 doğruluk)
- Açı bazlı performans:
  - 0°: %40.9 doğruluk
  - 90°: %34.1 doğruluk
  - 180°: %36.4 doğruluk
  - 270°: %31.8 doğruluk

### 4. Model Eğitimi (`train/`)

OpenCLIP modelini özel veri seti ile fine-tune eder.

**Eğitim Konfigürasyonu:**
```bash
python -m open_clip_train.main \
  --train-data "train.csv" \
  --val-data "val.csv" \
  --dataset-type "csv" \
  --csv-separator "," \
  --csv-img-key "image_path" \
  --csv-caption-key "model_name" \
  --model "ViT-L-14-quickgelu" \
  --pretrained "openai" \
  --batch-size 32 \
  --lr 5e-4 \
  --wd 0.01 \
  --epochs 100 \
  --warmup 500 \
  --workers 4 \
  --precision amp \
  --report-to "tensorboard" \
  --save-frequency 5 \
  --aug-cfg scale="0.9,1.1" color_jitter=0.3 \
  --name "cfv-vit-l14-improved3"
```

**Veri Seti İstatistikleri:**
- Eğitim: 1,202 örnek
- Doğrulama: 272 örnek
- Test: 602 örnek

**Tahmin Özellikleri:**
- 5 farklı prompt template kullanır
- Türkçe prompt desteği
- Görsel sonuç üretimi
- Güven eşiği: 0.90

## 📊 Sonuçlar ve Analizler

### Prompt Engineering Sonuçları
- En etkili prompt: "a front view photo of a BMW X3" (%21.93 güven)
- Ensemble yaklaşımı minimal iyileştirme sağlar (%0.0003)
- Detaylı prompt'lar genellikle daha iyi performans gösterir

### Model Karşılaştırma Sonuçları
- **En Hızlı**: ViT-B-32 (10.94ms inference)
- **En Doğru**: ViT-L-14 (100% güven)
- **En Büyük**: ViT-L-14 (427.6M parametre)
- **En Verimli**: ViT-B-32 (151.3M parametre, yüksek doğruluk)

### Zero-Shot Performansı
- **Genel Doğruluk**: %35.8
- **En İyi Sınıflar**: 8 sınıf %100 doğruluk
- **Açı Etkisi**: 0° açısında en iyi performans (%40.9)
- **Güven Ortalaması**: %2.38

## 📈 Görselleştirmeler

Proje aşağıdaki görselleştirmeleri üretir:
- `prompt_engineering_results.png`: Prompt template performansları
- `model_comparison.png`: Model karşılaştırma grafikleri
- `confusion_matrix.png`: Zero-shot sınıflandırma karışıklık matrisi
- `results/`: Her test görseli için tahmin sonuçları

## 🔧 Gereksinimler

```bash
pip install open_clip_torch torch torchvision
pip install pillow matplotlib seaborn pandas
pip install scikit-learn tqdm
```

## 📝 Notlar

1. **Model Seçimi**: ViT-B-32 genellikle hız-doğruluk dengesi için en iyi seçimdir
2. **Prompt Engineering**: Detaylı ve açıya özel prompt'lar daha iyi sonuç verir
3. **Zero-Shot**: Eğitim gerektirmez ancak sınırlı doğruluk sağlar
4. **Fine-tuning**: Özel veri seti ile eğitim daha yüksek doğruluk sağlar
