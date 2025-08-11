// Sonuç sayfası JavaScript işlevselliği
document.addEventListener('DOMContentLoaded', function() {
    const originalCanvas = document.getElementById('originalCanvas');
    const resultCanvases = document.querySelectorAll('.resultCanvas');

    // Sayfa yüklendiğinde resimleri çiz
    if (originalCanvas) {
        drawOriginalImage();
    }

    if (resultCanvases.length > 0) {
        drawResultImages();
    }

    /**
     * Orijinal resmi çizer
     */
    function drawOriginalImage() {
        try {
            const imageData = getOriginalImageData();
            if (imageData) {
                drawImage(originalCanvas, imageData, 280);
            }
        } catch (error) {
            console.error('Error drawing original image:', error);
        }
    }

    /**
     * Sonuç resimlerini çizer
     */
    function drawResultImages() {
        resultCanvases.forEach((canvas, index) => {
            try {
                const imageData = getResultImageData(canvas);
                if (imageData) {
                    drawImage(canvas, imageData, 140);
                }
            } catch (error) {
                console.error(`Error drawing result image ${index}:`, error);
            }
        });
    }

    /**
     * Orijinal resim verisini alır
     */
    function getOriginalImageData() {
        const scriptTag = document.getElementById('originalImageData');
        if (scriptTag) {
            return JSON.parse(scriptTag.getAttribute('data-image'));
        }
        return null;
    }

    /**
     * Sonuç resim verisini alır
     */
    function getResultImageData(canvas) {
        const imageData = canvas.getAttribute('data-image');
        return imageData ? JSON.parse(imageData) : null;
    }

    /**
     * Canvas'a resim çizer
     */
    // static/js/results.js dosyasını açın ve bu fonksiyonu güncelleyin
    function drawImage(canvas, imageData, size = 280) {
        if (!canvas || !imageData) return;

        const ctx = canvas.getContext('2d');
        
        // imageData'yı Float32Array'e çevir
        const tensor = new Float32Array(imageData);
        const data = new Uint8ClampedArray(tensor.length);
        
        // Tensor değerlerini 0-255 aralığına getir
        for (let i = 0; i < tensor.length; i++) {
            data[i] = Math.round(tensor[i] * 255);
        }
        
        // Geçici canvas oluştur (28x28)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // ImageData oluştur ve geçici canvas'a çiz
        const imgData = tempCtx.createImageData(28, 28);
        imgData.data.set(data);
        tempCtx.putImageData(imgData, 0, 0);
        
        // Hedef canvas'a büyüt (pikselleştirme için)
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, 28, 28, 0, 0, size, size);
        
        // Canvas arka planını beyaz yap
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(tempCanvas, 0, 0, 28, 28, 0, 0, size, size);
    }
});

// Yardımcı fonksiyonlar
window.ImageUtils = {
    /**
     * Base64'den canvas'a resim çizer
     */
    drawFromBase64: function(canvas, base64, size = 280) {
        const img = new Image();
        img.onload = function() {
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, size, size);
        };
        img.src = base64;
    },

    /**
     * Canvas'ı base64'e çevirir
     */
    toBase64: function(canvas) {
        return canvas.toDataURL();
    }
};