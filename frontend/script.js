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

    // Helper function to retry fetch
    async function fetchWithRetry(url, options, maxRetries = 2) {
        let retries = 0;
        while (retries <= maxRetries) {
            try {
                const response = await fetch(url, options);
                if (!response.ok) {
                    throw new Error(`Error: ${response.status} - ${response.statusText}`);
                }
                return response;
            } catch (error) {
                if (error.name === 'AbortError') {
                    throw error; // Don't retry timeouts
                }
                
                retries++;
                console.log(`Retry attempt ${retries}/${maxRetries}`);
                
                if (retries > maxRetries) {
                    throw error;
                }
                
                // Wait before retrying (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, 1000 * retries));
            }
        }
    }

    // Process image when process button is clicked
    processBtn.addEventListener('click', async () => {
        if (!uploadedFile) return;
        
        // Show loading spinner
        processedPreview.innerHTML = `
            <div class="processing-container">
                <div class="loading-spinner"></div>
                <p class="processing-text">Processing your image...</p>
                <p class="processing-subtext">This may take up to 30 seconds</p>
            </div>
        `;
        processBtn.disabled = true;
        
        // Create form data for the API request
        const formData = new FormData();
        formData.append('file', uploadedFile);

        // Create a controller to handle timeouts
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
        
        try {
            // Send request to backend API with timeout controller and retry
            const response = await fetchWithRetry('http://localhost:8000/process', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            const imageBlob = await response.blob();
            
            console.log('Received image blob:', imageBlob.size, 'bytes');
            
            // Create a new image object to ensure it's fully loaded
            const img = new Image();
            const imageUrl = URL.createObjectURL(imageBlob);
            
            // Show a loading message while the image loads
            processedPreview.innerHTML = `
                <div class="processing-container">
                    <div class="loading-spinner"></div>
                    <p class="processing-text">Loading processed image...</p>
                </div>
            `;
            
            // Set up promise for image loading
            const imageLoaded = new Promise((resolve, reject) => {
                img.onload = function() {
                    console.log('Image loaded successfully');
                    resolve();
                };
                
                img.onerror = function(err) {
                    console.error('Error loading image:', err);
                    reject(new Error('Failed to load the processed image'));
                };
            });
            
            // Set the source to trigger loading
            img.src = imageUrl;
            
            // Wait for image to load
            await imageLoaded;
            
            // Clear loading spinner and display processed image
            processedPreview.innerHTML = '';
            processedPreview.appendChild(img);
            
            // Enable download button
            downloadBtn.href = imageUrl;
            downloadBtn.style.display = 'inline-block';
            
        } catch (error) {
            clearTimeout(timeoutId);
            console.error('Error processing image:', error);
            
            // Show a more detailed error message
            let errorMessage = error.message;
            if (error.name === 'AbortError') {
                errorMessage = 'The request took too long to complete. Please try again or use a smaller image.';
            }
            
            processedPreview.innerHTML = `
                <div class="processing-container" style="color: #e74c3c;">
                    <i class="fas fa-exclamation-circle" style="font-size: 2rem; margin-bottom: 10px;"></i>
                    <p>Error processing image. Please try again.</p>
                    <p style="font-size: 0.8rem; color: #666;">${errorMessage}</p>
                </div>
            `;
        } finally {
            // Always re-enable the process button
            processBtn.disabled = false;
        }
    });

    // Function to handle files (validation and preview)
    function handleFiles(files) {
        const file = files[0];
        
        // Check if the file is an image
        if (!file.type.match('image.*')) {
            fileInfo.style.display = 'block';
            fileInfo.textContent = 'Please select an image file (PNG, JPG, JPEG, etc.)';
            processBtn.disabled = true;
            return;
        }
        
        // Disable the process button until preview is loaded
        processBtn.disabled = true;
        
        // Save the file for later use
        uploadedFile = file;
        
        // Display file info
        fileInfo.style.display = 'block';
        fileInfo.textContent = `File: ${file.name} (${formatFileSize(file.size)})`;
        
        // Show loading indicator in the original preview
        originalPreview.innerHTML = `
            <div class="processing-container">
                <div class="loading-spinner"></div>
                <p class="processing-text">Loading preview...</p>
            </div>
        `;
        
        // Reset processed image preview
        processedPreview.innerHTML = '<p>Waiting for processing</p>';
        downloadBtn.style.display = 'none';
        
        // Create preview using FileReader
        const reader = new FileReader();
        
        reader.onload = (e) => {
            // Pre-load the image to get dimensions and ensure it's valid
            const img = new Image();
            
            img.onload = function() {
                // Update UI with loaded preview
                originalPreview.innerHTML = '';
                originalPreview.appendChild(img);
                
                // Enable process button only after preview is successfully loaded
                processBtn.disabled = false;
                
                // Optional: Auto-click the process button if needed
                // processBtn.click();
            };
            
            img.onerror = function() {
                originalPreview.innerHTML = '<p>Failed to load image preview</p>';
                processBtn.disabled = true;
            };
            
            // Begin loading the image
            img.src = e.target.result;
        };
        
        reader.onerror = () => {
            originalPreview.innerHTML = '<p>Error loading image</p>';
            processBtn.disabled = true;
        };
        
        // Start reading the file as a data URL
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