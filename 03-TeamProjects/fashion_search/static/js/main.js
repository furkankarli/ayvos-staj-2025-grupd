// Ana sayfa JavaScript işlevselliği
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file-upload');
    const dropZone = fileInput.closest('.upload-area');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');

    // Form gönderimi
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    // Drag and drop
    if (dropZone) {
        setupDragAndDrop(dropZone, fileInput);
    }

    // Dosya seçimi
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    /**
     * Form gönderimini işler
     */
    async function handleFormSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        
        // Loading göster
        showLoading();
        hideError();
        
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.text();
            
            // Sonuç sayfasına yönlendir
            document.open();
            document.write(result);
            document.close();
            
        } catch (error) {
            console.error('Upload error:', error);
            showError('Dosya yüklenirken bir hata oluştu: ' + error.message);
            hideLoading();
        }
    }

    /**
     * Drag and drop ayarları
     */
    function setupDragAndDrop(zone, input) {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                handleFileSelect({ target: input });
            }
        });
    }

    /**
     * Dosya seçimini işler
     */
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            console.log('Selected file:', file.name);
            // Burada dosya önizleme eklenebilir
        }
    }

    /**
     * Loading göstergesini gösterir
     */
    function showLoading() {
        if (loading) {
            loading.classList.add('show');
        }
    }

    /**
     * Loading göstergesini gizler
     */
    function hideLoading() {
        if (loading) {
            loading.classList.remove('show');
        }
    }

    /**
     * Hata mesajını gösterir
     */
    function showError(message) {
        if (errorMessage) {
            errorMessage.querySelector('p').textContent = message;
            errorMessage.classList.add('show');
        }
    }

    /**
     * Hata mesajını gizler
     */
    function hideError() {
        if (errorMessage) {
            errorMessage.classList.remove('show');
        }
    }
});