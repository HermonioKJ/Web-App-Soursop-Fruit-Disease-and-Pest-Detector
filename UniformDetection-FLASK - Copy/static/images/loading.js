// loading.js
document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const loadingOverlay = document.getElementById('loading-overlay');

    form.addEventListener('submit', function () {
        loadingOverlay.style.display = 'block';
    });
});
