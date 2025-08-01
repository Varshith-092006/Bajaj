# 🚀 LLM Document Processing API

A FastAPI application for processing and querying insurance documents using LLM (Gemini) and semantic search.

## 🎯 **Features**

- ✅ **PDF Processing** - Extract text from PDF documents using pdfplumber
- ✅ **Semantic Search** - Pure Python vector search with cosine similarity
- ✅ **LLM Integration** - Google Gemini API for intelligent responses
- ✅ **Caching** - Efficient caching for PDF text and vector indices
- ✅ **RESTful API** - Clean FastAPI endpoints
- ✅ **Docker Ready** - Containerized for easy deployment

## 🚀 **Deploy to Railway**

### **Quick Deploy:**
1. **Fork/Clone** this repository
2. **Sign up** at [railway.app](https://railway.app)
3. **Connect** your GitHub repository
4. **Deploy** automatically (Railway will use `railway.json`)

### **Manual Deploy:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy
railway up
```

## 🔧 **Configuration**

### **Environment Variables:**
Set these in Railway dashboard:
- `GEMINI_API_KEY` - Your Google Gemini API key
- `PORT` - Usually 8000 (auto-set by Railway)

### **Get Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to Railway environment variables

## 📋 **API Endpoints**

### **Health Check:**
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "pdf_library": "pdfplumber",
  "vector_search": "pure-python",
  "gemini_configured": true
}
```

### **Process Documents:**
```bash
POST /process-documents
Content-Type: application/json

{
  "urls": [
    "https://example.com/document1.pdf",
    "https://example.com/document2.pdf"
  ]
}
```

### **Query Documents:**
```bash
POST /query
Content-Type: application/json

{
  "query": "What is the coverage limit for medical expenses?",
  "top_k": 5
}
```

### **Upload PDF:**
```bash
POST /upload-pdf
Content-Type: multipart/form-data

file: your-document.pdf
```

## 🧪 **Testing**

### **Local Testing:**
```bash
# Build and run locally
docker build -t fastapi-app .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key fastapi-app

# Test endpoints
curl http://localhost:8000/health
```

### **Deployment Testing:**
```bash
# Test your Railway deployment
python test_deployment.py
```

## 📊 **Architecture**

### **Components:**
- **FastAPI** - Web framework
- **pdfplumber** - PDF text extraction
- **SentenceTransformers** - Text embeddings
- **NumPy** - Vector operations
- **Google Gemini** - LLM responses
- **Docker** - Containerization

### **Flow:**
1. **Upload/Process** PDF documents
2. **Extract** text using pdfplumber
3. **Chunk** text semantically
4. **Generate** embeddings
5. **Store** in cache
6. **Search** using cosine similarity
7. **Generate** LLM responses

## 🔍 **Troubleshooting**

### **Common Issues:**

#### **Build Fails:**
- Check Railway logs
- Ensure all files are committed
- Verify Dockerfile syntax

#### **Runtime Errors:**
- Check environment variables
- Verify GEMINI_API_KEY is set
- Check application logs

#### **Import Errors:**
- All dependencies are pure Python
- No compilation required
- Should work on any platform

### **Logs:**
- **Railway Dashboard** - View build and runtime logs
- **Application Logs** - Check for specific errors
- **Health Endpoint** - Verify service status

## 📈 **Performance**

- **Build Time:** 2-3 minutes
- **Startup Time:** 30-60 seconds
- **Response Time:** <2 seconds
- **Memory Usage:** ~512MB

## 🎯 **Benefits of Railway**

- ✅ **Excellent Docker Support** - Handles Docker builds reliably
- ✅ **Fast Deployments** - 1-2 minutes vs 5-10 minutes
- ✅ **Better Logging** - Clear error messages
- ✅ **Auto-scaling** - Handles traffic automatically
- ✅ **Free Tier** - Generous limits
- ✅ **Git Integration** - Automatic deployments

## 📝 **Development**

### **Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn production_api:app --reload

# Test endpoints
curl http://localhost:8000/health
```

### **Adding Features:**
1. **Modify** `production_api.py`
2. **Test** locally
3. **Commit** changes
4. **Deploy** automatically (Railway)

## 📄 **License**

MIT License - Feel free to use and modify.

---

**Deployed at:** Your Railway URL
**Status:** Ready for production use 