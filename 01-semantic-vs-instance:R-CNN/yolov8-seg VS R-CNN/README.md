# YOLOv8-Seg ve Mask R-CNN ile Segmentasyon Karşılaştırması

## Summary
YOLOv8 ve Mask R-CNN ile segmentasyon karşılaştırması

## Description
Bu proje, YOLOv8-Seg ve Mask R-CNN modellerinin aynı görüntü üzerinde instance ve semantic segmentasyon performanslarını karşılaştırır. Kod, iki modelin çıktılarını yan yana görselleştirerek segmentasyon kalitesini ve nesne tespit başarımını analiz etmenizi sağlar.

## Kurulum ve Kullanım

1. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install ultralytics torch torchvision opencv-python matplotlib
    ```
2. `Segmentasyon/photo.jpg` yolunda bir test görseli bulundurun.
3. `yolov8n-seg.pt` dosyasını proje dizinine indirin (Ultralytics resmi sitesinden).
4. Python dosyasını çalıştırın:
    ```bash
    python yolov8-vs-Rcnn.py
    ```
5. Sonuçlar, iki modelin segmentasyon çıktılarının karşılaştırmalı görselleri olarak ekranda gösterilecektir.

## Sonuç
Bu çalışma sayesinde, YOLOv8-Seg ve Mask R-CNN modellerinin segmentasyon çıktıları doğrudan karşılaştırılabilir ve uygulamanız için en uygun
