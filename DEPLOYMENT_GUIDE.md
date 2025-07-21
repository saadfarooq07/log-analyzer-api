# Deployment Guide for Google Cloud Run

This guide walks you through deploying the Log Analyzer API to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: [Create one](https://cloud.google.com/free) if you don't have one
2. **Google Cloud Project**: Create a new project or use an existing one
3. **API Keys**:
   - [Google AI (Gemini) API Key](https://makersuite.google.com/app/apikey)
   - [Groq API Key](https://console.groq.com/keys)
   - [Tavily API Key](https://tavily.com/)

## Step 1: Prepare Your Repository

1. **Fork or Clone this Repository**
   ```bash
   git clone https://github.com/yourusername/log-analyzer-api.git
   cd log-analyzer-api
   ```

2. **Push to Your GitHub Account**
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/log-analyzer-api.git
   git push -u origin main
   ```

## Step 2: Deploy from Google Cloud Console

### A. Navigate to Cloud Run

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project
3. Navigate to **Cloud Run** from the menu

### B. Create Service

1. Click **"Create Service"**
2. Select **"Continuously deploy new revisions from a source repository"**
3. Click **"Set up with Cloud Build"**

### C. Connect Repository

1. **Source Repository**:
   - Provider: GitHub
   - Click **"Authenticate"** and authorize Google Cloud Build
   - Select your repository: `YOUR_USERNAME/log-analyzer-api`
   - Branch: `^main$` (or your default branch)

2. **Build Configuration**:
   - Build Type: **"Source"** (uses Buildpacks automatically)
   - No Dockerfile needed - Cloud Run detects Python automatically

### D. Configure Service

1. **Service Settings**:
   - Service name: `log-analyzer-api`
   - Region: Choose closest to your users (e.g., `us-central1`)
   - CPU allocation: **"CPU is only allocated during request processing"**
   - Autoscaling:
     - Minimum instances: `0`
     - Maximum instances: `100`

2. **Container Settings**:
   - Memory: `2 GiB`
   - CPU: `1`
   - Request timeout: `300` seconds
   - Container port: `8080` (automatically detected)

3. **Environment Variables**:
   Click **"Add Variable"** for each:
   - Name: `GEMINI_API_KEY`, Value: `your-gemini-api-key`
   - Name: `GROQ_API_KEY`, Value: `your-groq-api-key`
   - Name: `TAVILY_API_KEY`, Value: `your-tavily-api-key`

4. **Security**:
   - Authentication: **"Allow unauthenticated invocations"** (for public API)
   - Or select **"Require authentication"** for private use

5. Click **"Create"**

## Step 3: First Deployment

1. Cloud Build will automatically:
   - Detect Python from `requirements.txt`
   - Build a container using Google's buildpacks
   - Deploy to Cloud Run

2. Monitor the build:
   - Click on the build ID to see progress
   - First build takes 5-10 minutes
   - Subsequent builds are faster (2-3 minutes)

3. Once deployed, you'll get a URL like:
   ```
   https://log-analyzer-api-abc123-uc.a.run.app
   ```

## Step 4: Test Your Deployment

```bash
# Test health endpoint
curl https://your-service-url.run.app/health

# Test analysis
curl -X POST https://your-service-url.run.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "log_content": "ERROR: Test error message",
    "environment_details": {"os": "Linux"}
  }'
```

## Step 5: Set Up Continuous Deployment

Every push to your main branch will now:
1. Trigger Cloud Build
2. Build a new container
3. Deploy to Cloud Run
4. Automatically route traffic to the new version

## Optional: Using gcloud CLI

If you prefer command line deployment:

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Deploy from source
gcloud run deploy log-analyzer-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars "GEMINI_API_KEY=your-key,GROQ_API_KEY=your-key,TAVILY_API_KEY=your-key"
```

## Production Best Practices

### 1. Use Secret Manager for API Keys

```bash
# Create secrets
echo -n "your-gemini-key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your-groq-key" | gcloud secrets create groq-api-key --data-file=-
echo -n "your-tavily-key" | gcloud secrets create tavily-api-key --data-file=-

# Update service to use secrets
gcloud run services update log-analyzer-api \
  --update-secrets="GEMINI_API_KEY=gemini-api-key:latest,GROQ_API_KEY=groq-api-key:latest,TAVILY_API_KEY=tavily-api-key:latest"
```

### 2. Set Up Custom Domain

1. Go to Cloud Run service
2. Click **"Manage Custom Domains"**
3. Add your domain and follow verification steps

### 3. Enable Cloud Logging

```bash
# View logs
gcloud run services logs read log-analyzer-api --limit 50

# Stream logs
gcloud run services logs tail log-analyzer-api
```

### 4. Set Up Monitoring

1. Go to **Cloud Monitoring**
2. Create alerts for:
   - High error rate (> 5%)
   - High latency (> 5s)
   - Memory usage (> 80%)

### 5. Configure IAM for Private Access

```bash
# Remove public access
gcloud run services update log-analyzer-api --no-allow-unauthenticated

# Grant access to specific users
gcloud run services add-iam-policy-binding log-analyzer-api \
  --member="user:email@example.com" \
  --role="roles/run.invoker"
```

## Troubleshooting

### Build Fails

1. Check build logs in Cloud Build
2. Common issues:
   - Missing dependencies in requirements.txt
   - Python version mismatch (ensure runtime.txt specifies python-3.11)

### Service Won't Start

1. Check Cloud Run logs
2. Common issues:
   - Missing environment variables
   - Port mismatch (must use $PORT)

### High Latency

1. Increase memory allocation
2. Enable "CPU always allocated"
3. Set minimum instances to 1

### API Key Errors

1. Verify all three API keys are set
2. Check Secret Manager permissions
3. Ensure keys are valid and have quota

## Cost Estimation

For typical usage (1000 requests/day, 2MB average log):
- Compute: ~$5-10/month
- Network: ~$1/month
- Total: ~$6-11/month

Cloud Run only charges for actual usage, so costs scale with traffic.

## Next Steps

1. **Add Authentication**: Implement API keys or OAuth
2. **Add Caching**: Use Memorystore for Redis
3. **Add Database**: Store analysis results in Firestore
4. **Add Queue**: Use Cloud Tasks for async processing
5. **Add CDN**: Use Cloud CDN for static assets

## Support

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Python on Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)
- [Buildpacks Documentation](https://cloud.google.com/docs/buildpacks/overview)