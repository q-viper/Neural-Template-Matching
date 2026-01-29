# Neural Template Matching Web App 🎯

A web-based template matching application using ONNX.js for real-time neural network inference directly in your browser.

## 🚀 Live Demo

**GitHub Pages**: [https://q-viper.github.io/Neural-Template-Matching/](https://q-viper.github.io/Neural-Template-Matching/)

## ✨ Features

- 🧠 **Real-time Neural Template Matching** using custom ONNX model
- 🌐 **No Server Required** - runs entirely in browser with ONNX.js  
- 📱 **Responsive Design** - works on desktop and mobile
- 🎨 **Interactive Visualization** - real-time threshold adjustment
- ✂️ **Smart Cropping** - extracts matched regions with transparency
- ⚡ **Fast Inference** - WebAssembly-powered execution

## 🎮 How to Use

1. **Visit the live demo** at the GitHub Pages link above
2. **Upload a main image** - the image to search in
3. **Upload a template image** - what you want to find
4. **Click "🔍 Run Template Matching"** - see real-time results
5. **Adjust threshold** - fine-tune matching sensitivity
6. **View results** - see confidence mask and cropped regions

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **ML Runtime**: ONNX.js v1.19.0 with WebAssembly backend
- **Neural Network**: Custom UNet-based template matching model
- **Hosting**: GitHub Pages (static site)
- **Model Format**: ONNX (Open Neural Network Exchange)

## 📁 Project Structure

```
Neural-Template-Matching/
├── index.html              # Main web application
├── app.js                  # JavaScript application logic
├── model.onnx             # ONNX neural network model
├── model_instructions.md  # Model deployment guide
├── temp_matching/         # Python training code
├── notebooks/             # Jupyter notebooks
├── assets/                # Training data and results
└── README.md              # This file
```

## 🔧 Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/q-viper/Neural-Template-Matching.git
   cd Neural-Template-Matching
   ```

2. **Add your model** (see `model_instructions.md` for options):
   ```bash
   # Option 1: Copy your model to root directory
   cp path/to/your/model.onnx ./model.onnx
   ```

3. **Open in browser**:
   - Simply open `index.html` in Chrome, Firefox, or Edge
   - Or use a local server: `python -m http.server 8000`

## 🎯 Model Information

- **Input Shape**: `[1, 2, 3, 512, 512]` (batch, images, channels, height, width)
- **Output**: Confidence mask `[1, 1, 512, 512]` 
- **Architecture**: Custom UNet with multiplication encoding
- **Training**: PyTorch → ONNX conversion
- **Size**: ~174MB (optimized for inference)

## 🌐 Deployment Options

### GitHub Pages (Current)
- ✅ Free hosting on GitHub
- ✅ Automatic deployment from main branch
- ✅ Custom domain support
- ❌ Large file limitations (>100MB)

### Alternative Model Hosting
- **Google Drive**: Convert share link to direct download
- **Dropbox**: Change `?dl=0` to `?dl=1` 
- **Hugging Face**: ML model repository hosting
- **CDN**: Use any CDN service for large files

## 🐛 Troubleshooting

### Model Not Loading
1. **Check browser console** (F12) for error messages
2. **Verify model path** - ensure model.onnx is accessible
3. **CORS issues** - use raw GitHub URLs or proper hosting
4. **File size** - GitHub has 100MB file limit

### Performance Issues  
1. **Use modern browser** (Chrome 57+, Firefox 52+, Safari 11+)
2. **Enable WebAssembly** in browser settings
3. **Close other tabs** to free up memory
4. **Consider model optimization** for mobile devices

### No Results Displayed
1. **Upload both images** before running inference
2. **Adjust threshold** - try values between 0.1-0.9
3. **Check image formats** - supports JPEG, PNG, WebP
4. **Try different templates** - some may work better than others

## 🚀 Getting Started (Quick)

Want to try it right now?

1. Go to: **[https://q-viper.github.io/Neural-Template-Matching/](https://q-viper.github.io/Neural-Template-Matching/)**
2. Upload any image as "main image"
3. Upload a cropped portion of that image as "template" 
4. Click "🔍 Run Template Matching"
5. Watch the magic happen! ✨

## 📚 Research & Development

This project implements neural template matching using deep learning. For technical details:

- **Training Code**: See `temp_matching/` directory
- **Model Architecture**: Custom UNet implementation
- **Evaluation Notebooks**: See `notebooks/` directory  
- **Results & Visualizations**: See `assets/` directory

## 📄 License

MIT License - feel free to use for research and commercial projects.

## 🤝 Contributing

Contributions welcome! Please feel free to:
- Report bugs and issues
- Suggest new features  
- Submit pull requests
- Share your results and use cases

---

**Built with ❤️ by [q-viper](https://github.com/q-viper)**

*Neural Template Matching - Making computer vision accessible in the browser*
