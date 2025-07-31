# Deployment Guide for Render

## Prerequisites
- Render account
- Gemini API key from Google AI Studio

## Deployment Steps

### 1. Prepare Your Repository
- Ensure all files are committed to your Git repository
- Make sure `production_api.py` is your main application file
- Verify `requirements.txt` contains all necessary dependencies

### 2. Deploy to Render

1. **Connect Repository**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" and select "Web Service"
   - Connect your Git repository

2. **Configure Service**
   - **Name**: `fastapi-insurance-app` (or your preferred name)
   - **Runtime**: Python 3
   - **Build Command**: Leave empty (handled by render.yaml)
   - **Start Command**: Leave empty (handled by render.yaml)
   - **Plan**: Choose appropriate plan (Free tier works for testing)

3. **Environment Variables**
   - Add `GEMINI_API_KEY` with your actual API key
   - The `PORT` variable is automatically set by Render

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### 3. Verify Deployment

1. **Check Health Endpoint**
   - Visit `https://your-app-name.onrender.com/health`
   - Should return status "healthy"

2. **Test API Endpoints**
   - Root: `https://your-app-name.onrender.com/`
   - Query: `https://your-app-name.onrender.com/api/v1/query`
   - Run: `https://your-app-name.onrender.com/api/v1/hackrx/run`

## Troubleshooting

### Common Issues

1. **PyMuPDF Compilation Failures**
   - **Problem**: PyMuPDF fails to compile during build
   - **Solution**: The current configuration includes all necessary system dependencies
   - **Fallback**: If PyMuPDF still fails, use `production_api_fallback.py` with pdfplumber

2. **Build Failures**
   - Check Render logs for specific error messages
   - Ensure all dependencies are in `requirements.txt`
   - Verify system packages are installed in build command

3. **Runtime Errors**
   - Check application logs in Render dashboard
   - Verify `GEMINI_API_KEY` is set correctly
   - Ensure all required files are present

4. **Memory Issues**
   - Reduce worker count in gunicorn configuration
   - Optimize model loading and caching

### PyMuPDF Specific Issues

If you encounter PyMuPDF compilation errors, try these solutions:

1. **Use Fallback Version**
   ```bash
   # Rename the fallback version to main
   mv production_api_fallback.py production_api.py
   ```

2. **Update Requirements**
   ```bash
   # Use the fallback requirements
   cp requirements_fallback.txt requirements.txt
   ```

3. **Alternative Build Command**
   ```yaml
   buildCommand: |
     apt-get update && apt-get install -y \
       libmupdf-dev python3-dev pkg-config build-essential \
       gcc g++ make cmake
     export MU_PDF_USE_SYSTEM_LIBS=1
     pip install --no-cache-dir -r requirements.txt
   ```

### Logs and Monitoring
- View logs in Render dashboard under your service
- Monitor resource usage and performance
- Set up alerts for downtime

## API Usage

### Health Check
```bash
curl https://your-app-name.onrender.com/health
```

### Query Documents
```bash
curl -X POST https://your-app-name.onrender.com/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is covered under this policy?",
    "documents": ["https://example.com/policy.pdf"],
    "top_k": 5
  }'
```

### Run Multiple Questions
```bash
curl -X POST https://your-app-name.onrender.com/api/v1/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/policy.pdf"],
    "questions": ["What is covered?", "What are exclusions?"]
  }'
```

## Security Notes
- Never commit API keys to your repository
- Use environment variables for sensitive data
- Consider rate limiting for production use
- Monitor API usage and costs

## Alternative Deployment Options

### Option 1: Use pdfplumber (More Stable)
If PyMuPDF continues to fail, use the fallback version:

1. Replace `production_api.py` with `production_api_fallback.py`
2. Use `requirements_fallback.txt` instead of `requirements.txt`
3. Deploy normally

### Option 2: Use Pre-built PyMuPDF
```bash
# Add to requirements.txt
PyMuPDF-binary==1.23.7
```

### Option 3: Use Docker
Create a Dockerfile for more control over the build environment. 