# 🎯 Model Deployment Instructions

## Option 1: Upload model.onnx to Repository Root (Recommended)

1. **Add model.onnx to root directory**:
   ```
   temp_matching/
   ├── index.html
   ├── app.js  
   ├── model.onnx          ← Add your model here
   └── ...
   ```

2. **Temporarily allow large files**:
   ```bash
   # Remove *.onnx from .gitignore temporarily
   git add model.onnx
   git commit -m "Add ONNX model for GitHub Pages"
   git push
   ```

3. **Model will be accessible at**: 
   `https://raw.githubusercontent.com/q-viper/Neural-Template-Matching/main/model.onnx`

## Option 2: Use External Hosting

### Google Drive (Free, Easy)
1. Upload model.onnx to Google Drive
2. Right-click → Get Link → Anyone with link can view
3. Convert sharing URL to direct download URL:
   ```
   From: https://drive.google.com/file/d/FILE_ID/view?usp=sharing  
   To:   https://drive.google.com/uc?export=download&id=FILE_ID
   ```
4. Update app.js with your Google Drive URL

### Dropbox (Alternative)
1. Upload to Dropbox
2. Generate sharing link
3. Change `?dl=0` to `?dl=1` in URL

### Hugging Face (ML-focused, Free)
1. Create account at huggingface.co
2. Create new model repository
3. Upload model.onnx
4. Use direct link: `https://huggingface.co/username/repo-name/resolve/main/model.onnx`

## Current App Configuration

The app now tries these URLs in order:
1. ✅ `https://raw.githubusercontent.com/q-viper/Neural-Template-Matching/main/model.onnx` (GitHub raw)
2. ❌ `https://github.com/q-viper/Neural-Template-Matching/releases/download/v0.0.1/model.onnx` (CORS blocked)
3. ✅ `../model.onnx` (local development)
4. ✅ `./model.onnx` (same directory)
5. ✅ `model.onnx` (relative path)

## Quick Fix for Testing

To get it working immediately:

```bash
# 1. Copy your model to root
cp app/model.onnx ./model.onnx

# 2. Temporarily remove from gitignore
sed -i '/\*.onnx/d' .gitignore

# 3. Add and commit
git add model.onnx
git commit -m "Add model for GitHub Pages demo"
git push

# 4. Restore gitignore (optional)
echo "*.onnx" >> .gitignore
git add .gitignore
git commit -m "Restore gitignore"
git push
```

Your app will then work at: `https://q-viper.github.io/Neural-Template-Matching/`
