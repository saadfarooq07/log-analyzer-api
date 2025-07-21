# API Documentation

## Base URL

```
https://your-service-url.run.app
```

## Authentication

Currently, the API supports two authentication modes:

1. **No Authentication** (default for development)
2. **Google Cloud IAM** (recommended for production)

To enable IAM authentication:
```bash
gcloud run services update log-analyzer-api --no-allow-unauthenticated
```

Then include the authorization header:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://your-service-url.run.app/api/v1/analyze
```

## Endpoints

### 1. Analyze Logs

Analyze log content and return structured results.

**Endpoint:** `POST /api/v1/analyze`

**Request Body:**
```json
{
  "log_content": "string (required)",
  "environment_details": {
    "os": "string",
    "service_version": "string",
    "additional_context": "any"
  },
  "application_name": "string (optional)",
  "analysis_type": "general|security|performance|error (default: general)",
  "include_suggestions": "boolean (default: true)",
  "include_documentation": "boolean (default: true)"
}
```

**Response:**
```json
{
  "analysis_id": "uuid",
  "timestamp": "ISO 8601 datetime",
  "status": "completed",
  "issues": [
    {
      "type": "string",
      "description": "string",
      "severity": "critical|high|medium|low",
      "line_number": "integer (optional)",
      "timestamp": "string (optional)",
      "pattern": "string (optional)"
    }
  ],
  "suggestions": [
    {
      "issue_type": "string",
      "suggestion": "string",
      "priority": "immediate|high|medium|low",
      "estimated_impact": "string (optional)"
    }
  ],
  "documentation_references": [
    {
      "title": "string",
      "url": "string",
      "relevance": "high|medium|low",
      "excerpt": "string (optional)"
    }
  ],
  "diagnostic_commands": [
    {
      "command": "string",
      "description": "string",
      "platform": "linux|windows|macos (optional)"
    }
  ],
  "summary": "string (optional)",
  "metrics": {
    "total_lines": "integer",
    "issues_found": "integer",
    "processing_time": "float (seconds)",
    "log_size_mb": "float"
  }
}
```

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid request data
- `413 Payload Too Large`: Log file exceeds size limit
- `500 Internal Server Error`: Analysis failed
- `504 Gateway Timeout`: Analysis timeout

**Example:**
```bash
curl -X POST https://your-service-url.run.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "log_content": "2024-01-20 ERROR: Database connection failed\n2024-01-20 ERROR: Retry attempt 1 failed",
    "environment_details": {
      "os": "Ubuntu 22.04",
      "postgresql_version": "14.5"
    },
    "application_name": "web-api",
    "analysis_type": "general"
  }'
```

### 2. Stream Analysis

Analyze logs with real-time progress updates using Server-Sent Events.

**Endpoint:** `POST /api/v1/analyze/stream`

**Request Body:** Same as `/analyze`

**Response:** Server-Sent Events stream

**Event Types:**
1. **Progress Event**
   ```json
   {
     "type": "progress",
     "data": {
       "status": "starting|analyzing|completing",
       "message": "string",
       "progress": "integer (0-100)"
     }
   }
   ```

2. **Partial Result Event**
   ```json
   {
     "type": "partial_result",
     "data": {
       "issues_found": "integer",
       "message": "string"
     }
   }
   ```

3. **Complete Event**
   ```json
   {
     "type": "complete",
     "data": {
       // Same as /analyze response
     }
   }
   ```

4. **Error Event**
   ```json
   {
     "type": "error",
     "data": {
       "error": "string",
       "message": "string"
     }
   }
   ```

**Example (JavaScript):**
```javascript
const eventSource = new EventSource('/api/v1/analyze/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    log_content: logData,
    environment_details: { os: 'Linux' }
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'progress':
      console.log(`Progress: ${data.data.progress}%`);
      break;
    case 'complete':
      console.log('Analysis complete:', data.data);
      eventSource.close();
      break;
    case 'error':
      console.error('Error:', data.data.message);
      eventSource.close();
      break;
  }
};
```

### 3. Health Check

Check if the service is healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "log-analyzer-api",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy

### 4. Readiness Check

Check if the service is ready to handle requests.

**Endpoint:** `GET /ready`

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "api_keys": "configured",
    "models": "available"
  }
}
```

**Status Codes:**
- `200 OK`: Service is ready
- `503 Service Unavailable`: Service is not ready

### 5. Root Endpoint

Get basic API information.

**Endpoint:** `GET /`

**Response:**
```json
{
  "name": "Log Analyzer API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "api": "/api/v1"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limits:**
  - 100 requests per minute per IP
  - 10 concurrent requests per IP
  - 100MB total data per hour per IP

- **Headers returned:**
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "ERROR_TYPE",
  "message": "Human-readable error message",
  "details": {
    "additional": "context"
  },
  "timestamp": "2024-01-20T15:30:45.123Z"
}
```

**Common Error Types:**
- `VALIDATION_ERROR`: Invalid request data
- `SIZE_LIMIT_ERROR`: Log file too large
- `TIMEOUT_ERROR`: Analysis timeout
- `API_KEY_ERROR`: Invalid or missing API key
- `INTERNAL_ERROR`: Server error

## Best Practices

1. **Log Size**: Keep logs under 10MB for best performance
2. **Timeout**: Set client timeout to at least 60 seconds
3. **Retries**: Implement exponential backoff for retries
4. **Streaming**: Use streaming endpoint for logs > 1MB
5. **Compression**: Gzip compress large requests

## SDK Examples

### Python SDK
```python
import requests
from typing import Dict, Any

class LogAnalyzerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def analyze(self, log_content: str, **kwargs) -> Dict[str, Any]:
        """Analyze log content."""
        url = f"{self.base_url}/api/v1/analyze"
        data = {
            "log_content": log_content,
            **kwargs
        }
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check service health."""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

# Usage
client = LogAnalyzerClient("https://your-service-url.run.app")
result = client.analyze(
    log_content="ERROR: Database connection failed",
    environment_details={"os": "Linux"}
)
```

### Node.js SDK
```javascript
const axios = require('axios');

class LogAnalyzerClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }
  
  async analyze(logContent, options = {}) {
    const url = `${this.baseUrl}/api/v1/analyze`;
    const { data } = await axios.post(url, {
      log_content: logContent,
      ...options
    }, {
      timeout: 60000
    });
    return data;
  }
  
  async healthCheck() {
    try {
      const { data } = await axios.get(`${this.baseUrl}/health`);
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

// Usage
const client = new LogAnalyzerClient('https://your-service-url.run.app');
const result = await client.analyze(
  'ERROR: Database connection failed',
  { environment_details: { os: 'Linux' } }
);
```