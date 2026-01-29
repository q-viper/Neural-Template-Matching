// Global state
let mainImage = null;
let templateImage = null;
let session = null;
let modelLoaded = false;

const DOM = {
    mainImageInput: document.getElementById('mainImageInput'),
    templateImageInput: document.getElementById('templateImageInput'),
    mainImagePreview: document.getElementById('mainImagePreview'),
    templateImagePreview: document.getElementById('templateImagePreview'),
    runButton: document.getElementById('runButton'),
    status: document.getElementById('status'),
    canvasSection: document.getElementById('canvasSection'),
    inputCanvasContainer: document.getElementById('inputCanvasContainer'),
    settingsPanel: document.getElementById('settingsPanel'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    // Info panel
    infoModel: document.getElementById('infoModel'),
    infoInputShape: document.getElementById('infoInputShape'),
    infoTime: document.getElementById('infoTime'),
    infoThreshold: document.getElementById('infoThreshold'),
};

// Utilities
function showStatus(message, type = 'loading') {
    DOM.status.textContent = '';
    if (type === 'loading') {
        DOM.status.innerHTML = `<span class="spinner"></span>${message}`;
    } else {
        DOM.status.textContent = message;
    }
    DOM.status.className = `status show ${type}`;
}

function hideStatus() {
    DOM.status.classList.remove('show');
}

// Loading overlay functions
function showLoadingOverlay(message = '🔄 Running neural network inference...') {
    console.log('Attempting to show loading overlay:', message);
    console.log('Loading overlay element:', DOM.loadingOverlay);
    console.log('Loading text element:', DOM.loadingText);
    
    if (DOM.loadingText) {
        DOM.loadingText.textContent = message;
    }
    if (DOM.loadingOverlay) {
        DOM.loadingOverlay.classList.add('show');
    }
    document.body.style.overflow = 'hidden'; // Prevent scrolling
    
    console.log('Loading overlay should now be visible');
}

function hideLoadingOverlay() {
    console.log('Hiding loading overlay');
    if (DOM.loadingOverlay) {
        DOM.loadingOverlay.classList.remove('show');
    }
    document.body.style.overflow = ''; // Restore scrolling
}

// Test function to verify loading overlay works
function testLoadingOverlay() {
    console.log('Testing loading overlay...');
    showLoadingOverlay('🧪 Testing loading overlay...');
    setTimeout(() => {
        hideLoadingOverlay();
        console.log('Test complete');
    }, 3000);
}

// File input handling for separate images
DOM.mainImageInput.addEventListener('change', (e) => handleImageSelect(e, 'main'));
DOM.templateImageInput.addEventListener('change', (e) => handleImageSelect(e, 'template'));

function handleImageSelect(event, imageType) {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous results when new image is selected
    DOM.canvasSection.classList.remove('show');
    hideStatus();
    window.lastOutput = null;
    window.lastOutputShape = null;

    readImageFile(file).then((img) => {
        if (imageType === 'main') {
            mainImage = img;
            displayImagePreview(img, DOM.mainImagePreview, '🖼️ Main Image');
        } else {
            templateImage = img;
            displayImagePreview(img, DOM.templateImagePreview, '🔍 Template');
        }
        
        checkImagesReady();
        showStatus(`✅ ${imageType === 'main' ? 'Main' : 'Template'} image loaded - Ready to run inference`, 'success');
    }).catch((err) => {
        showStatus(`Error loading ${imageType} image: ${err.message}`, 'error');
    });
}

function displayImagePreview(img, container, label) {
    container.innerHTML = '';
    container.className = 'preview-container has-image';
    
    const imgElement = document.createElement('img');
    imgElement.src = img.src;
    imgElement.alt = label;
    
    container.appendChild(imgElement);
}

function checkImagesReady() {
    const bothImagesReady = mainImage && templateImage;
    DOM.runButton.disabled = !bothImagesReady;
    DOM.runButton.textContent = bothImagesReady ? '🔍 Run Template Matching' : '🔍 Run Template Matching';
    DOM.settingsPanel.style.display = bothImagesReady ? 'block' : 'none';
    
    if (bothImagesReady) {
        // Don't display input canvases here - only show them after inference
        // Clear previous inference info when new images are loaded
        DOM.infoInputShape.textContent = '-';
        DOM.infoTime.textContent = '-';
    } else {
        // Hide results when images are not ready
        DOM.canvasSection.classList.remove('show');
    }
}

function readImageFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = e.target.result;
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsDataURL(file);
    });
}

function displayInputCanvases() {
    DOM.inputCanvasContainer.innerHTML = '';
    const images = [mainImage, templateImage];
    const labels = ['🖼️ Main Image', '🔍 Template'];
    
    images.forEach((img, idx) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'canvas-wrapper';

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const label = document.createElement('div');
        label.className = 'canvas-label';
        label.textContent = labels[idx];

        wrapper.appendChild(canvas);
        wrapper.appendChild(label);
        DOM.inputCanvasContainer.appendChild(wrapper);
    });
}

// Slider listener with real-time update
DOM.thresholdSlider.addEventListener('input', (e) => {
    const thresholdValue = parseFloat(e.target.value).toFixed(2);
    DOM.thresholdValue.textContent = thresholdValue;
    DOM.infoThreshold.textContent = thresholdValue;
    
    // If inference has been run and we have output data, update visualization in real-time
    if (window.lastOutput && window.lastOutputShape) {
        visualizeOutput(window.lastOutput, window.lastOutputShape);
    }
});

// Model loading & inference
async function loadModel(forceReload = false) {
    // Try multiple paths for the model file to work with GitHub Pages
    const modelPaths = [
        './model.onnx',
        'model.onnx',
        '/model.onnx',
        window.location.href + 'model.onnx'
    ];
    
    try {
        // If forcing reload or session doesn't exist, recreate it
        if (forceReload || !session) {
            showStatus('Loading ONNX model...', 'loading');
            showLoadingOverlay('📦 Loading ONNX model...');
            
            // Dispose of existing session if it exists
            if (session) {
                try {
                    session = null;
                } catch (e) {
                    console.warn('Error disposing session:', e);
                }
            }
        } else if (modelLoaded && session) {
            return true; // Model already loaded and working
        }

        // Configure execution provider with error handling
        ort.env.wasm.simdEnabled = true;
        ort.env.wasm.multithreadEnabled = true;
        ort.env.logLevel = 'warning'; // Reduce log verbosity

        // Add error event listeners
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && event.reason.message && event.reason.message.includes('message channel closed')) {
                console.warn('ONNX Runtime: Message channel closed warning (non-critical)');
                event.preventDefault(); // Prevent the error from showing in console
            }
        });

        // Try loading the model from different paths
        let loadError = null;
        for (const modelPath of modelPaths) {
            try {
                console.log(`Trying to load model from: ${modelPath}`);
                session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'basic',
                    enableCpuMemArena: false,
                    enableMemPattern: false,
                    executionMode: 'sequential',
                    interOpNumThreads: 1,
                    intraOpNumThreads: 1,
                });
                console.log(`Successfully loaded model from: ${modelPath}`);
                break; // Success, exit loop
            } catch (err) {
                console.warn(`Failed to load model from ${modelPath}:`, err.message);
                loadError = err;
                continue; // Try next path
            }
        }

        if (!session) {
            throw new Error(`Failed to load model from any path. Last error: ${loadError?.message || 'Unknown error'}`);
        }

        modelLoaded = true;
        DOM.infoModel.textContent = 'model.onnx';
        showStatus('✅ Model loaded successfully - Ready for GitHub Pages!', 'success');
        hideLoadingOverlay(); // Hide loading overlay when called directly
        return true;
    } catch (err) {
        showStatus(`Model load error: ${err.message}. For GitHub Pages, ensure model.onnx is in the repository root.`, 'error');
        console.error('ONNX Model Error:', err);
        hideLoadingOverlay(); // Hide loading overlay on error
        return false;
    }
}

async function runInference() {
    console.log('runInference called');
    
    if (!modelLoaded) {
        const loaded = await loadModel();
        if (!loaded) return;
    }

    if (!mainImage || !templateImage) {
        showStatus('Upload both main and template images first', 'error');
        return;
    }

    console.log('About to show loading overlay and disable button');
    
    // Disable the run button and show loading state
    DOM.runButton.disabled = true;
    DOM.runButton.textContent = '⏳ Processing...';
    
    // Show loading overlay
    showLoadingOverlay('🔄 Running neural network inference...');
    
    // Also make sure status is shown for fallback
    showStatus('🔄 Running neural network inference...', 'loading');
    
    // Add a small delay to ensure UI updates
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        console.log('Starting inference...');

        // Preprocess images: [main image, template]
        console.log('Preprocessing images...');
        DOM.loadingText.textContent = '📊 Preprocessing images...';
        const { input, shape } = preprocessImages([mainImage, templateImage]);
        DOM.infoInputShape.textContent = `[${shape.join(', ')}]`;

        // Create tensor
        console.log('Creating tensor...');
        DOM.loadingText.textContent = '🔧 Creating tensors...';
        const inputTensor = new ort.Tensor('float32', input, shape);
        const feeds = { combined_input: inputTensor };

        // Run inference with timeout
        console.log('Running inference...');
        DOM.loadingText.textContent = '🧠 Running neural network...';
        const startTime = performance.now();
        
        // Add timeout to prevent hanging
        const inferencePromise = session.run(feeds);
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Inference timeout after 30 seconds')), 30000);
        });
        
        const results = await Promise.race([inferencePromise, timeoutPromise]);
        const endTime = performance.now();
        console.log('Inference completed');
        const inferenceTime = (endTime - startTime).toFixed(2);

        DOM.infoTime.textContent = `${inferenceTime}ms`;

        // Get output and store for real-time threshold updates
        console.log('Processing results...');
        DOM.loadingText.textContent = '📈 Processing results...';
        const output = results.masks.data;
        const outputShape = results.masks.dims;
        window.lastOutput = output;
        window.lastOutputShape = outputShape;

        // Dispose of tensors to prevent memory leaks
        console.log('Disposing tensors...');
        inputTensor.dispose();
        results.masks.dispose();

        // Post-process and visualize
        console.log('Visualizing output...');
        DOM.loadingText.textContent = '🎨 Generating visualizations...';
        visualizeOutput(output, outputShape);
        console.log('Visualization completed');

        showStatus(`✅ Inference complete in ${inferenceTime}ms - Upload new images to run again`, 'success');
        
        // Re-enable the run button and hide loading overlay
        DOM.runButton.disabled = false;
        DOM.runButton.textContent = '🔍 Run Template Matching';
        hideLoadingOverlay();
    } catch (err) {
        showStatus(`Inference error: ${err.message}`, 'error');
        console.error('Inference Error:', err);
        
        // If there's a timeout or session error, reset the model
        if (err.message.includes('timeout') || err.message.includes('session')) {
            console.log('Resetting model due to error...');
            modelLoaded = false;
            session = null;
        }
        
        // Re-enable the run button on error and hide loading overlay
        DOM.runButton.disabled = false;
        DOM.runButton.textContent = '🔍 Run Template Matching';
        hideLoadingOverlay();
    }
}

function preprocessImages(images) {
    const height = 512;
    const width = 512;
    const channels = 3;

    // Input shape: [1, 2, 3, 512, 512] → [batch, 2 images, channels, h, w]
    const input = new Float32Array(1 * 2 * channels * height * width);

    // Process each image (image + template)
    for (let imgIdx = 0; imgIdx < Math.min(images.length, 2); imgIdx++) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        const img = images[imgIdx];
        // Resize image to fit canvas while maintaining aspect ratio
        const scale = Math.min(width / img.width, height / img.height);
        const x = (width - img.width * scale) / 2;
        const y = (height - img.height * scale) / 2;
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        // Calculate base offset for this image: [imgIdx * channels * height * width]
        const imgOffset = imgIdx * channels * height * width;

        // Normalize to [0, 1] and arrange in NCHW format
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i] / 255.0;
            const g = data[i + 1] / 255.0;
            const b = data[i + 2] / 255.0;

            const pixelIdx = i / 4;
            
            // Place in correct position: [batch, img, channel, h, w]
            input[imgOffset + 0 * height * width + pixelIdx] = r; // Red channel
            input[imgOffset + 1 * height * width + pixelIdx] = g; // Green channel
            input[imgOffset + 2 * height * width + pixelIdx] = b; // Blue channel
        }
    }

    // If only 1 image, duplicate it as the second image
    if (images.length === 1) {
        const firstImgData = input.subarray(0, channels * height * width);
        input.set(firstImgData, channels * height * width);
    }

    return { input, shape: [1, 2, channels, height, width] };
}

function visualizeOutput(output, shape) {
    const [batch, channels, height, width] = shape;
    const threshold = parseFloat(DOM.thresholdSlider.value);

    // Find min and max values for better visualization
    let minVal = output[0];
    let maxVal = output[0];
    for (let i = 1; i < output.length; i++) {
        if (output[i] < minVal) minVal = output[i];
        if (output[i] > maxVal) maxVal = output[i];
    }

    // Clear and setup the canvas container for side-by-side display
    DOM.inputCanvasContainer.innerHTML = '';

    // Create output mask canvas
    const maskWrapper = document.createElement('div');
    maskWrapper.className = 'canvas-wrapper';
    
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 300; // Fixed width for consistent display
    maskCanvas.height = 200; // Fixed height for consistent display
    const maskCtx = maskCanvas.getContext('2d');
    
    // Create a temporary canvas at original size for processing
    const tempMaskCanvas = document.createElement('canvas');
    tempMaskCanvas.width = width;
    tempMaskCanvas.height = height;
    const tempMaskCtx = tempMaskCanvas.getContext('2d');
    
    const maskImageData = tempMaskCtx.createImageData(width, height);
    const maskData = maskImageData.data;

    // Render the mask
    for (let i = 0; i < height * width; i++) {
        const rawValue = output[i];
        const normalizedValue = maxVal > minVal ? (rawValue - minVal) / (maxVal - minVal) : 0;
        const isMatch = rawValue >= threshold;

        if (isMatch) {
            // Hot colors for matches (red-yellow-white)
            const intensity = Math.round(normalizedValue * 255);
            maskData[i * 4] = 255; // Red
            maskData[i * 4 + 1] = Math.round(intensity * 0.8); // Green
            maskData[i * 4 + 2] = Math.round(intensity * 0.3); // Blue
        } else {
            // Cool colors for non-matches (blue-purple-black)
            const intensity = Math.round(normalizedValue * 128);
            maskData[i * 4] = Math.round(intensity * 0.2); // Red
            maskData[i * 4 + 1] = Math.round(intensity * 0.3); // Green  
            maskData[i * 4 + 2] = Math.min(255, intensity + 64); // Blue
        }
        maskData[i * 4 + 3] = Math.round(200 + normalizedValue * 55); // Alpha
    }

    tempMaskCtx.putImageData(maskImageData, 0, 0);
    
    // Draw the processed mask to the display canvas at fixed size
    maskCtx.drawImage(tempMaskCanvas, 0, 0, width, height, 0, 0, 300, 200);

    const maskLabel = document.createElement('div');
    maskLabel.className = 'canvas-label';
    maskLabel.textContent = '🎯 Output Mask';

    maskWrapper.appendChild(maskCanvas);
    maskWrapper.appendChild(maskLabel);

    // Create matched regions canvas (cropped region from original image)
    const matchWrapper = document.createElement('div');
    matchWrapper.className = 'canvas-wrapper';
    
    const matchCanvas = document.createElement('canvas');
    const matchCtx = matchCanvas.getContext('2d');

    // Find bounding box of matched regions
    let minX = width, maxX = 0, minY = height, maxY = 0;
    let hasMatches = false;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const rawValue = output[idx];
            const isMatch = rawValue >= threshold;

            if (isMatch) {
                hasMatches = true;
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }
    }

    if (hasMatches && mainImage) {
        // Add padding to the bounding box
        const padding = 20;
        minX = Math.max(0, minX - padding);
        maxX = Math.min(width - 1, maxX + padding);
        minY = Math.max(0, minY - padding);
        maxY = Math.min(height - 1, maxY + padding);

        const cropWidth = maxX - minX + 1;
        const cropHeight = maxY - minY + 1;

        // Set canvas size to crop dimensions
        matchCanvas.width = cropWidth;
        matchCanvas.height = cropHeight;

        // Calculate scaling and positioning for the main image
        const scale = Math.min(width / mainImage.width, height / mainImage.height);
        const offsetX = (width - mainImage.width * scale) / 2;
        const offsetY = (height - mainImage.height * scale) / 2;

        // Calculate source coordinates on the original image
        const srcX = (minX - offsetX) / scale;
        const srcY = (minY - offsetY) / scale;
        const srcWidth = cropWidth / scale;
        const srcHeight = cropHeight / scale;

        // Ensure source coordinates are within image bounds
        const clampedSrcX = Math.max(0, Math.min(mainImage.width - 1, srcX));
        const clampedSrcY = Math.max(0, Math.min(mainImage.height - 1, srcY));
        const clampedSrcWidth = Math.min(srcWidth, mainImage.width - clampedSrcX);
        const clampedSrcHeight = Math.min(srcHeight, mainImage.height - clampedSrcY);

        // First, draw the main image to a temporary canvas to get pixel data
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = cropWidth;
        tempCanvas.height = cropHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw the cropped region from main image
        if (clampedSrcWidth > 0 && clampedSrcHeight > 0) {
            tempCtx.drawImage(
                mainImage,
                clampedSrcX, clampedSrcY, clampedSrcWidth, clampedSrcHeight,
                0, 0, cropWidth, cropHeight
            );

            // Get the image data from the cropped region
            const imageData = tempCtx.getImageData(0, 0, cropWidth, cropHeight);
            const data = imageData.data;

            // Apply mask: set background to transparent where mask is 0
            for (let y = 0; y < cropHeight; y++) {
                for (let x = 0; x < cropWidth; x++) {
                    const globalX = minX + x;
                    const globalY = minY + y;
                    const pixelIdx = y * cropWidth + x;
                    
                    if (globalX < width && globalY < height) {
                        const maskIdx = globalY * width + globalX;
                        const maskValue = output[maskIdx];
                        
                        // If mask value is very low (close to 0), make pixel transparent
                        if (maskValue < 0.01) { // Threshold for "background"
                            data[pixelIdx * 4 + 3] = 0; // Set alpha to 0 (transparent)
                        }
                        // For areas with some mask value, keep original alpha but can adjust
                        else if (maskValue < threshold) {
                            // Slightly reduce opacity for low confidence areas
                            data[pixelIdx * 4 + 3] = Math.round(data[pixelIdx * 4 + 3] * 0.3);
                        }
                        // High confidence areas keep full opacity
                    } else {
                        // Outside bounds, make transparent
                        data[pixelIdx * 4 + 3] = 0;
                    }
                }
            }

            // Put the modified image data to the main canvas
            matchCtx.putImageData(imageData, 0, 0);
        }
    } else {
        // No matches found, show a placeholder
        matchCanvas.width = width;
        matchCanvas.height = height;
        matchCtx.fillStyle = '#f0f0f0';
        matchCtx.fillRect(0, 0, width, height);
        matchCtx.fillStyle = '#999';
        matchCtx.font = '16px Arial';
        matchCtx.textAlign = 'center';
        matchCtx.fillText('No matches found', width / 2, height / 2);
    }

    const matchLabel = document.createElement('div');
    matchLabel.className = 'canvas-label';
    matchLabel.textContent = '✂️ Cropped Match Region';

    matchWrapper.appendChild(matchCanvas);
    matchWrapper.appendChild(matchLabel);

    // Add both canvases to container
    DOM.inputCanvasContainer.appendChild(maskWrapper);
    DOM.inputCanvasContainer.appendChild(matchWrapper);
    
    DOM.canvasSection.classList.add('show');
    
    // Log stats for debugging
    console.log(`Output stats - Min: ${minVal.toFixed(4)}, Max: ${maxVal.toFixed(4)}, Threshold: ${threshold}`);
}

function clearAll() {
    mainImage = null;
    templateImage = null;
    window.lastOutput = null;
    window.lastOutputShape = null;
    DOM.mainImageInput.value = '';
    DOM.templateImageInput.value = '';
    DOM.runButton.disabled = true;
    DOM.runButton.textContent = '🔍 Run Template Matching';
    DOM.settingsPanel.style.display = 'none';
    DOM.canvasSection.classList.remove('show');
    DOM.inputCanvasContainer.innerHTML = '';
    
    // Clear preview containers
    DOM.mainImagePreview.innerHTML = '<div class="preview-placeholder">Main image preview</div>';
    DOM.mainImagePreview.className = 'preview-container';
    DOM.templateImagePreview.innerHTML = '<div class="preview-placeholder">Template preview</div>';
    DOM.templateImagePreview.className = 'preview-container';
    
    hideStatus();

    // Reset info panel
    DOM.infoModel.textContent = '-';
    DOM.infoInputShape.textContent = '-';
    DOM.infoTime.textContent = '-';
    DOM.infoThreshold.textContent = '0.50';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking elements...');
    console.log('Loading overlay found:', !!DOM.loadingOverlay);
    console.log('Loading text found:', !!DOM.loadingText);
    console.log('Run button found:', !!DOM.runButton);
    
    DOM.infoThreshold.textContent = '0.50';
    
    // Initialize preview containers with placeholder text
    DOM.mainImagePreview.innerHTML = '<div class="preview-placeholder">Main image preview</div>';
    DOM.templateImagePreview.innerHTML = '<div class="preview-placeholder">Template preview</div>';
});