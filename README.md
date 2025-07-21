# Log Analyzer API

A FastAPI-based service that analyzes log files using LangGraph and AI models (Gemini and Groq). Designed for easy deployment to Google Cloud Run.

## Features

### Core Features
- 🔍 **Intelligent Log Analysis**: Identifies errors, warnings, and patterns in log files
- 🚀 **Fast Processing**: Optimized for logs up to 10MB
- 📊 **Structured Output**: Returns issues, suggestions, and diagnostic commands
- 📚 **Documentation Search**: Finds relevant documentation for identified issues
- 🌊 **Streaming Support**: Server-Sent Events for real-time analysis updates
- ☁️ **Cloud Run Ready**: Optimized for serverless deployment

### Enhanced Features (NEW)
- 💬 **Interactive Mode**: Q&A flow for clarification during analysis
- 💾 **Memory/Persistence**: Analysis history and context retention
- 🔄 **Advanced Cycle Detection**: Prevents infinite loops with pattern recognition
- 🛡️ **Circuit Breaker**: Fault tolerance for external services
- ⏱️ **API Rate Limiting**: Prevents quota exhaustion
- 🎯 **Specialized Analyzers**: Domain-specific analysis (HDFS, Security, Application)
- 🚄 **Advanced Streaming**: Parallel chunk processing for large logs
- 📈 **Resource Tracking**: Memory and CPU monitoring
- 🗄️ **Intelligent Caching**: Performance optimization with LRU cache

See [ENHANCED_FEATURES.md](./ENHANCED_FEATURES.md) for detailed documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- API Keys:
  - [Google AI (Gemini)](https://makersuite.google.com/app/apikey)
  - [Groq](https://console.groq.com/keys)
  - [Tavily](https://tavily.com/)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/log-analyzer-api.git
   cd log-analyzer-api
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## API Endpoints

### Analyze Logs
```bash
POST /api/v1/analyze
Content-Type: application/json

{
  "log_content": "2024-01-20 ERROR: Database connection failed...",
  "environment_details": {
    "os": "Ubuntu 22.04",
    "service": "PostgreSQL 14"
  },
  "application_name": "web-api",
  "analysis_type": "general"
}
```

### Stream Analysis (SSE)
```bash
POST /api/v1/analyze/stream
Content-Type: application/json

# Same request body as /analyze
# Returns Server-Sent Events stream
```

## Deployment Options

### Option 1: LangGraph Cloud SaaS (Recommended)

The easiest way to deploy - fully managed by LangChain with GitHub integration.

**Quick Start:**
1. Push your code to GitHub
2. Go to [smith.langchain.com](https://smith.langchain.com)
3. Click "LangGraph" → "New Deployment"
4. Connect your GitHub repo
5. Add your API keys
6. Deploy!

See [CLOUD_SAAS_DEPLOYMENT.md](./CLOUD_SAAS_DEPLOYMENT.md) for detailed instructions.

### Option 2: Self-Hosted Options

- **Google Cloud Run**: See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Standalone Container**: See [LANGGRAPH_CLOUD_DEPLOYMENT.md](./LANGGRAPH_CLOUD_DEPLOYMENT.md)

## Option 2: Google Cloud Run Deployment

### Option 1: Deploy from Source (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/log-analyzer-api.git
   git push -u origin main
   ```

2. **Deploy from Cloud Console**
   - Go to [Cloud Run](https://console.cloud.google.com/run)
   - Click "Create Service"
   - Select "Continuously deploy from a repository"
   - Connect your GitHub account and select the repository
   - Configure:
     - Service name: `log-analyzer-api`
     - Region: Your preferred region
     - Authentication: Allow unauthenticated invocations (or configure as needed)
     - Container port: 8080
     - Memory: 2 GiB
     - CPU: 1
     - Request timeout: 300 seconds
     - Environment variables:
       ```
       GEMINI_API_KEY=your-key
       GROQ_API_KEY=your-key
       TAVILY_API_KEY=your-key
       ```

### Option 2: Deploy with gcloud CLI

```bash
# Install gcloud CLI if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy directly from source
gcloud run deploy log-analyzer-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars "GEMINI_API_KEY=your-key,GROQ_API_KEY=your-key,TAVILY_API_KEY=your-key"
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google AI API key for Gemini model | Yes | - |
| `GROQ_API_KEY` | Groq API key for orchestration | Yes | - |
| `TAVILY_API_KEY` | Tavily API key for documentation search | Yes | - |
| `LOG_LEVEL` | Logging level | No | INFO |
| `MAX_LOG_SIZE_MB` | Maximum log file size in MB | No | 10 |
| `ENABLE_STREAMING` | Enable SSE streaming endpoint | No | true |
| `ANALYSIS_TIMEOUT` | Analysis timeout in seconds | No | 300 |

### Using Secret Manager (Recommended for Production)

```bash
# Create secrets
echo -n "your-gemini-key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your-groq-key" | gcloud secrets create groq-api-key --data-file=-
echo -n "your-tavily-key" | gcloud secrets create tavily-api-key --data-file=-

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Update service to use secrets
gcloud run services update log-analyzer-api \
  --update-secrets="GEMINI_API_KEY=gemini-api-key:latest,GROQ_API_KEY=groq-api-key:latest,TAVILY_API_KEY=tavily-api-key:latest"
```

## Example Usage

### Python
```python
import requests

url = "https://your-service-url.run.app/api/v1/analyze"
data = {
    "log_content": """
    2024-01-20 10:15:23 ERROR [database] Connection timeout after 30s
    2024-01-20 10:15:24 ERROR [database] Failed to connect to PostgreSQL
    2024-01-20 10:15:25 WARN [api] Fallback to cache due to database error
    """,
    "environment_details": {
        "os": "Ubuntu 22.04",
        "postgresql_version": "14.5"
    },
    "application_name": "web-api"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Found {len(result['issues'])} issues")
for issue in result['issues']:
    print(f"- {issue['severity']}: {issue['description']}")
```

### cURL
```bash
curl -X POST https://your-service-url.run.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "log_content": "ERROR: Database connection failed",
    "environment_details": {"os": "Linux"}
  }'
```

## Response Format

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-20T15:30:45.123Z",
  "status": "completed",
  "issues": [
    {
      "type": "database_error",
      "description": "PostgreSQL connection timeout",
      "severity": "critical",
      "line_number": 1,
      "timestamp": "2024-01-20 10:15:23"
    }
  ],
  "suggestions": [
    {
      "issue_type": "database_error",
      "suggestion": "Check PostgreSQL service status and network connectivity",
      "priority": "high",
      "estimated_impact": "Restore database connectivity"
    }
  ],
  "documentation_references": [
    {
      "title": "PostgreSQL Connection Troubleshooting",
      "url": "https://www.postgresql.org/docs/current/runtime-config-connection.html",
      "relevance": "high",
      "excerpt": "Connection timeout parameters..."
    }
  ],
  "diagnostic_commands": [
    {
      "command": "systemctl status postgresql",
      "description": "Check PostgreSQL service status",
      "platform": "linux"
    }
  ],
  "summary": "Critical database connectivity issue detected",
  "metrics": {
    "total_lines": 3,
    "issues_found": 1,
    "processing_time": 2.34,
    "log_size_mb": 0.001
  }
}
```

## Monitoring

### Cloud Run Metrics
- View metrics in Cloud Console: CPU, Memory, Request count, Latency
- Set up alerts for errors or high latency

### Logging
```bash
# View logs
gcloud run services logs read log-analyzer-api --limit 50

# Stream logs
gcloud run services logs tail log-analyzer-api
```

## Cost Optimization

- Cloud Run charges only for actual usage
- Typical costs:
  - CPU: ~$0.00002400 per vCPU-second
  - Memory: ~$0.00000250 per GiB-second
  - Requests: ~$0.40 per million requests
- Set minimum instances to 0 for development
- Use Cloud Scheduler for warming if needed

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all three API keys are set correctly
   - Check Secret Manager permissions if using secrets

2. **Timeout Errors**
   - Increase Cloud Run timeout (max 3600 seconds)
   - Reduce log size or use streaming endpoint

3. **Memory Errors**
   - Increase Cloud Run memory allocation
   - Current limit: 10MB per log file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details