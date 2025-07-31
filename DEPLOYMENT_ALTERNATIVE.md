# Alternative Deployment Guide (No Rust Compilation)

## Problem
The main deployment is failing due to Rust compilation issues with faiss-cpu on Render's read-only file system.

## Solution
Use the scikit-learn alternative version that avoids Rust compilation entirely.

## Quick Fix

### Option 1: Use Alternative Version (Recommended)
```bash
# Replace the main files with the alternative version
cp production_api_alternative.py production_api.py
cp requirements_alternative.txt requirements.txt
```

### Option 2: Manual File Changes

1. **Update requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
pdfplumber==0.10.3
scikit-learn==1.3.0
sentence-transformers==2.2.2
google-generativeai==0.3.2
requests==2.31.0
aiofiles==23.2.1
gunicorn==21.2.0
numpy==1.24.3
huggingface_hub==0.19.4
```

2. **Update production_api.py:**
   - Replace faiss imports with scikit-learn
   - Replace `build_faiss_index_optimized` with `build_vector_index_sklearn`
   - Replace `search_chunks_enhanced` with `search_chunks_sklearn`

## Benefits of Alternative Version

✅ **No Rust compilation** - Uses pure Python libraries
✅ **Faster deployment** - No complex build process
✅ **More reliable** - Fewer dependencies
✅ **Same functionality** - All features preserved

## Performance Comparison

| Feature | FAISS Version | Scikit-learn Version |
|---------|---------------|---------------------|
| Build Time | ~5-10 minutes | ~2-3 minutes |
| Memory Usage | Higher | Lower |
| Search Speed | Faster | Slightly slower |
| Deployment Reliability | Lower | Higher |

## Deploy

After making the changes, deploy normally:

1. Commit the changes
2. Push to your repository
3. Render will automatically redeploy

## Verify Deployment

Check the health endpoint:
```bash
curl https://your-app-name.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "pdf_library": "pdfplumber",
  "vector_search": "scikit-learn"
}
```

## Troubleshooting

If you still encounter issues:

1. **Clear Render cache** - Delete and recreate the service
2. **Use Docker** - Create a Dockerfile for more control
3. **Try different Python version** - Use Python 3.11 instead of 3.13

## Rollback

If you need to go back to the FAISS version:
```bash
git checkout HEAD~1
git push
``` 