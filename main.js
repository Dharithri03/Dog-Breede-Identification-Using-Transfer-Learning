const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const uploadContent = document.getElementById('upload-content');
const submitBtn = document.getElementById('submit-btn');
const form = document.getElementById('upload-form');

// Trigger file input when clicking drop zone
if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());
}

// Handle File Input Change
if (fileInput) {
    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            showPreview(this.files[0]);
        }
    });
}

// Drag and Drop Logic
if (dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files && files.length > 0) {
        // Assign to file input for form submission
        fileInput.files = files;
        showPreview(files[0]);
    }
}

function showPreview(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
    }
    reader.readAsDataURL(file);
}

// Form Submit Validation
if (form) {
    form.addEventListener('submit', function (e) {
        if (!fileInput.files || fileInput.files.length === 0) {
            e.preventDefault();
            alert('Please upload an image first!');
        } else {
            submitBtn.innerHTML = '<span class="loader"></span> Analyzing...';
            submitBtn.style.opacity = '0.8';
        }
    });
}
