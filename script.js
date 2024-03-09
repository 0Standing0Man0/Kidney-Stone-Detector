function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const filePath = URL.createObjectURL(file);
            const link = document.createElement('a');
            link.href = filePath;
            link.download = file.name;
            link.click();
            URL.revokeObjectURL(filePath);
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select an image to upload.');
    }
}
