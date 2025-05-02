document.addEventListener('DOMContentLoaded', () => {
    // Mobile Navigation Toggle
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');

    burger.addEventListener('click', () => {
        nav.classList.toggle('nav-active');
        
        // Burger Animation
        burger.classList.toggle('toggle');
    });

    // Smooth Scrolling for Navigation Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust for fixed navbar
                    behavior: 'smooth'
                });
                
                // Mobile menu close after click
                if (nav.classList.contains('nav-active')) {
                    nav.classList.remove('nav-active');
                    burger.classList.remove('toggle');
                }
                
                // Update active link
                document.querySelectorAll('.nav-links a').forEach(link => {
                    link.classList.remove('active');
                });
                this.classList.add('active');
            }
        });
    });

    // Contact Form Submission
    const contactForm = document.getElementById('contactForm');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // In a real application, you would send this data to a server
            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                message: document.getElementById('message').value
            };
            
            // For demo purposes, just show an alert
            alert('Thanks for your message! We will get back to you soon.');
            contactForm.reset();
        });
    }

    // Rain Removal Demo Section
    const imageUpload = document.getElementById('imageUpload');
    const uploadArea = document.getElementById('uploadArea');
    const fileInfo = document.getElementById('fileInfo');
    const processBtn = document.getElementById('processBtn');
    const originalPreview = document.getElementById('originalPreview');
    const processedPreview = document.getElementById('processedPreview');
    const downloadBtn = document.getElementById('downloadBtn');
    
    let uploadedFile = null;

    // Handle drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('active');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Handle file input change
    uploadArea.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    });

    // Process image when process button is clicked
    processBtn.addEventListener('click', () => {
        if (!uploadedFile) return;
        
        // Show loading state
        processedPreview.innerHTML = '<p>Processing...</p>';
        processBtn.disabled = true;
        
        // Create form data for the API request
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        // Send request to backend API
        fetch('http://localhost:8000/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(imageBlob => {
            // Display processed image
            const imageUrl = URL.createObjectURL(imageBlob);
            processedPreview.innerHTML = '';
            
            const img = document.createElement('img');
            img.src = imageUrl;
            processedPreview.appendChild(img);
            
            // Enable download button
            downloadBtn.href = imageUrl;
            downloadBtn.style.display = 'inline-block';
            
            // Re-enable process button
            processBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error processing image:', error);
            processedPreview.innerHTML = '<p>Error processing image. Please try again.</p>';
            processBtn.disabled = false;
        });
    });

    // Function to handle files (validation and preview)
    function handleFiles(files) {
        const file = files[0];
        
        // Check if the file is an image
        if (!file.type.match('image.*')) {
            fileInfo.style.display = 'block';
            fileInfo.textContent = 'Please select an image file (PNG, JPG, JPEG, etc.)';
            return;
        }
        
        // Save the file for later use
        uploadedFile = file;
        
        // Display file info
        fileInfo.style.display = 'block';
        fileInfo.textContent = `File: ${file.name} (${formatFileSize(file.size)})`;
        
        // Create preview
        const reader = new FileReader();
        
        reader.onload = (e) => {
            originalPreview.innerHTML = '';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            originalPreview.appendChild(img);
            
            // Reset processed image preview
            processedPreview.innerHTML = '<p>Waiting for processing</p>';
            downloadBtn.style.display = 'none';
            
            // Enable process button
            processBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }

    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Scroll Spy for Navigation
    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY + 100; // Adjust for navbar
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('.nav-links a');
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                const id = section.getAttribute('id');
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });
}); 