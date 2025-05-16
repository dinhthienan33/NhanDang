document.addEventListener('DOMContentLoaded', () => {
    // Mobile Navigation Toggle
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    
    // Model status variables
    let modelReady = false;
    let modelType = 'unknown';
    let deviceType = 'unknown';

    // Check backend status on page load
    checkModelStatus();

    // Create fullscreen overlay elements for both images
    const fullscreenOverlay = document.createElement('div');
    fullscreenOverlay.className = 'fullscreen-overlay';
    
    const fullscreenImage = document.createElement('img');
    fullscreenOverlay.appendChild(fullscreenImage);
    
    const closeButton = document.createElement('div');
    closeButton.className = 'fullscreen-close';
    closeButton.innerHTML = '<i class="fas fa-times"></i>';
    fullscreenOverlay.appendChild(closeButton);
    
    document.body.appendChild(fullscreenOverlay);
    
    // Close fullscreen when clicking on overlay or close button
    fullscreenOverlay.addEventListener('click', () => {
        fullscreenOverlay.classList.remove('active');
    });
    
    closeButton.addEventListener('click', (e) => {
        e.stopPropagation();
        fullscreenOverlay.classList.remove('active');
    });

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
    const modelStatusIndicator = document.createElement('div');
    modelStatusIndicator.className = 'model-status';
    modelStatusIndicator.innerHTML = '<span class="loading-dot"></span> Checking model status...';
    document.querySelector('.demo-controls').prepend(modelStatusIndicator);
    
    let uploadedFile = null;

    // Handle drag and drop
    let uploadInProgress = false; // Flag to prevent duplicate uploads
    
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
        
        if (!uploadInProgress && e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Handle file input change
    // uploadArea.addEventListener('click', () => {
    //     if (!uploadInProgress) {
    //         imageUpload.click();
    //     }
    // });

    imageUpload.addEventListener('change', (e) => {
        if (!uploadInProgress && e.target.files.length) {
            handleFiles(e.target.files);
            // Auto-start processing
            // if (!processBtn.disabled) {
            //     processBtn.click();
            // }
        }
    });

    // Add fullscreen preview functionality
    function setupFullscreenPreview(previewElement) {
        previewElement.addEventListener('click', (e) => {
            // Make sure we're clicking on an image, not just the container
            if (e.target.tagName === 'IMG') {
                fullscreenImage.src = e.target.src;
                fullscreenOverlay.classList.add('active');
            }
        });
    }
    
    // Set up fullscreen for both preview containers
    setupFullscreenPreview(originalPreview);
    setupFullscreenPreview(processedPreview);

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

    // Helper function to check model status
    async function checkModelStatus() {
        try {
            const response = await fetch('http://localhost:8000/status');
            if (response.ok) {
                const data = await response.json();
                modelReady = data.model_ready;
                modelType = data.model_type;
                deviceType = data.device;
                
                // Update UI based on model status
                if (modelReady) {
                    modelStatusIndicator.innerHTML = `<span class="status-dot ready"></span> Model ready (${modelType} on ${deviceType})`;
                    modelStatusIndicator.classList.add('ready');
                    processBtn.disabled = !uploadedFile;
                } else {
                    modelStatusIndicator.innerHTML = `<span class="status-dot not-ready"></span> Using fallback model (${modelType})`;
                    modelStatusIndicator.classList.add('not-ready');
                    processBtn.disabled = !uploadedFile;
                }
            } else {
                modelStatusIndicator.innerHTML = '<span class="status-dot error"></span> Error connecting to backend';
                modelStatusIndicator.classList.add('error');
                processBtn.disabled = true;
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            modelStatusIndicator.innerHTML = '<span class="status-dot error"></span> Error connecting to backend';
            modelStatusIndicator.classList.add('error');
            processBtn.disabled = true;
            
            // Retry after 5 seconds
            setTimeout(checkModelStatus, 5000);
        }
    }

    // Process image when process button is clicked
    processBtn.addEventListener('click', async () => {
        if (!uploadedFile || uploadInProgress) return;
        
        uploadInProgress = true; // Set flag to prevent duplicate uploads
        
        // Disable process button while processing
        processBtn.disabled = true;
        
        // Show loading spinner
        processedPreview.innerHTML = `
            <div class="processing-container">
                <div class="loading-spinner"></div>
                <p class="processing-text">Processing your image...</p>
                <p class="processing-subtext">This may take up to 30 seconds</p>
            </div>
        `;
        
        // Create form data for the API request
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('upscale', 'true'); // Enable upscaling

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
            
            // Check if we got an error response with error message
            if (!response.ok) {
                let errorText = 'Error processing image';
                try {
                    const errorData = await response.json();
                    errorText = errorData.detail || errorText;
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                throw new Error(errorText);
            }
            
            const imageBlob = await response.blob();
            
            // Check if we received a valid image
            if (imageBlob.size === 0 || imageBlob.type.indexOf('image/') !== 0) {
                throw new Error('Invalid response from server - not an image');
            }
            
            console.log('Received image blob:', imageBlob.size, 'bytes', imageBlob.type);
            
            // Check if this is a fallback result
            const isFallback = response.headers.get('X-Fallback-Result') === 'true';
            
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
            processedPreview.classList.add('has-image');
            
            // Add notice if fallback model was used
            if (isFallback) {
                const fallbackNotice = document.createElement('div');
                fallbackNotice.className = 'fallback-notice';
                fallbackNotice.innerHTML = 'Notice: Fallback model was used for processing';
                processedPreview.appendChild(fallbackNotice);
            }
            
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
                    <p>Error processing image</p>
                    <p style="font-size: 0.8rem; color: #666;">${errorMessage}</p>
                    <button id="retryBtn" class="btn btn-secondary" style="margin-top: 10px;">Try Again</button>
                </div>
            `;
            
            // Add retry functionality
            const retryBtn = document.getElementById('retryBtn');
            if (retryBtn) {
                retryBtn.addEventListener('click', () => {
                    // Make sure we're not in processing state before retrying
                    if (!uploadInProgress) {
                        processBtn.click();
                    }
                });
            }
        } finally {
            // Always re-enable the process button and reset flag
            processBtn.disabled = false;
            uploadInProgress = false;
        }
    });

    function handleFiles(files) {
        if (uploadInProgress) return; // Prevent handling files while upload is in progress
        
        uploadInProgress = true; // Set flag immediately to prevent duplicate uploads
        
        if (files.length) {
            uploadedFile = files[0];
            
            // Check if the file is an image
            if (!uploadedFile.type.startsWith('image/')) {
                fileInfo.textContent = 'Please upload an image file.';
                fileInfo.classList.add('error');
                processBtn.disabled = true;
                uploadInProgress = false; // Reset flag on error
                return;
            }
            
            // Update the file info
            fileInfo.textContent = `${uploadedFile.name} (${formatFileSize(uploadedFile.size)})`;
            fileInfo.classList.remove('error');
            
            // Enable the process button only if model is ready
            processBtn.disabled = false;
            
            // Show the original image preview with loading indicator
            fileInfo.innerHTML = `<span class="loading-dot"></span> Loading preview: ${uploadedFile.name} (${formatFileSize(uploadedFile.size)})`;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                // Create a new image to get dimensions
                const img = new Image();
                img.onload = function() {
                    // Show dimensions in the file info
                    fileInfo.textContent = `${uploadedFile.name} (${formatFileSize(uploadedFile.size)}, ${img.width}x${img.height})`;
                    
                    // Now that we know the image loaded successfully, show it
                    originalPreview.innerHTML = `<img src="${e.target.result}" alt="Original Image">`;
                    originalPreview.classList.add('has-image');
                    
                    // Make the process button more prominent if model is ready
                    if (!processBtn.disabled) {
                        processBtn.classList.add('ready');
                    }
                    
                    uploadInProgress = false; // Reset flag only after image is fully loaded
                };
                
                img.onerror = function() {
                    fileInfo.textContent = `Error loading ${uploadedFile.name}`;
                    fileInfo.classList.add('error');
                    processBtn.disabled = true;
                    originalPreview.innerHTML = `<p>Failed to preview image</p>`;
                    uploadInProgress = false; // Reset flag on error
                };
                
                img.src = e.target.result;
            };
            
            reader.onerror = () => {
                fileInfo.textContent = `Error reading ${uploadedFile.name}`;
                fileInfo.classList.add('error');
                processBtn.disabled = true;
                uploadInProgress = false; // Reset flag on error
            };
            
            reader.readAsDataURL(uploadedFile);
            
            // Clear the processed preview
            processedPreview.innerHTML = '';
            processedPreview.classList.remove('has-image');
            
            // Hide the download button
            downloadBtn.style.display = 'none';
        } else {
            uploadInProgress = false; // Reset flag if no files
        }
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