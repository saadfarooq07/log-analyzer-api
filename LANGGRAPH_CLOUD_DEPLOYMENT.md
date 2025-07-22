# LangGraph Cloud Deployment Guide

This guide explains how to deploy the Log Analyzer API to LangGraph Cloud, which provides a managed infrastructure for LangGraph applications.

## Prerequisites

1. **LangSmith Account**: Sign up at [smith.langchain.com](https://smith.langchain.com)
2. **LangGraph Cloud Access**: Request access or check availability
3. **GitHub Repository Secrets** (configured in repository settings):
   - `GEMINI_API_KEY` - Google Gemini API key
   - `GROQ_API_KEY` - Groq API key  
   - `TAVILY_API_KEY` - Tavily search API key
   - `LANGCHAIN_API_KEY` - LangChain API key for LangGraph Cloud

## Option 1: Deploy via LangGraph CLI

### Step 1: Install LangGraph CLI

```bash
pip install -U langgraph-cli
```

### Step 2: Authenticate

```bash
langgraph auth login
```

### Step 3: Deploy from Repository

```bash
# From the repository root
langgraph deploy --name log-analyzer-api
```

The CLI will:
1. Read `langgraph.json` configuration
2. Package your application
3. Upload to LangGraph Cloud
4. Provide you with an API endpoint

### Step 4: Set Environment Variables

The environment variables are automatically configured from GitHub repository secrets during deployment. The following variables are required:

```bash
GEMINI_API_KEY=${GEMINI_API_KEY}
GROQ_API_KEY=${GROQ_API_KEY}
TAVILY_API_KEY=${TAVILY_API_KEY}
LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
USE_IMPROVED_LOG_ANALYZER=true
```

These are automatically set by the GitHub Actions workflow from repository secrets.

## Option 2: Deploy via GitHub Integration

### Step 1: Connect Repository

1. Go to [LangGraph Cloud Console](https://cloud.langchain.com)
2. Click "New Deployment"
3. Select "Deploy from GitHub"
4. Authorize and select your repository

### Step 2: Configure Deployment

1. **Deployment Name**: `log-analyzer-api`
2. **Branch**: `main`
3. **Config File**: `langgraph.json` (auto-detected)
4. **Environment Variables**:
   - Add your API keys in the UI

### Step 3: Deploy

Click "Deploy" and LangGraph Cloud will:
- Build your application
- Run tests (if configured)
- Deploy to managed infrastructure
- Provide API endpoints

## Using the Deployed API

Once deployed, you'll get endpoints like:

```
https://your-deployment.langgraph.cloud/
```

### Invoke the Graph

```python
import httpx
from langgraph_sdk import get_client

# Initialize client
client = get_client(url="https://your-deployment.langgraph.cloud")

# Run the graph
thread = await client.threads.create()
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="log_analyzer",
    input={
        "log_content": "ERROR: Database connection failed",
        "environment_details": {"os": "Linux"},
        "analysis_type": "general"
    }
)

# Get results
result = await client.runs.get(thread["thread_id"], run["run_id"])
print(result["analysis_result"])
```

### Using REST API

```bash
# Create a thread
curl -X POST https://your-deployment.langgraph.cloud/threads \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-langsmith-api-key"

# Run analysis
curl -X POST https://your-deployment.langgraph.cloud/threads/{thread_id}/runs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-langsmith-api-key" \
  -d '{
    "assistant_id": "log_analyzer",
    "input": {
      "log_content": "ERROR: Database connection failed",
      "environment_details": {"os": "Linux"}
    }
  }'
```

## Streaming Support

LangGraph Cloud automatically supports streaming:

```python
# Stream events
async for event in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="log_analyzer",
    input={...}
):
    print(event)
```

## Monitoring and Debugging

### View Traces in LangSmith

All runs are automatically traced in LangSmith:
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project
3. View traces, latency, and errors

### Check Logs

```bash
langgraph logs --deployment log-analyzer-api --tail
```

### View Metrics

```bash
langgraph metrics --deployment log-analyzer-api
```

## Scaling and Performance

LangGraph Cloud automatically handles:
- **Auto-scaling**: Based on request volume
- **Load balancing**: Across multiple instances
- **Caching**: For repeated analyses
- **Rate limiting**: To prevent abuse

### Configuration Options

In `langgraph.json`, you can add:

```json
{
  "dependencies": ["."],
  "graphs": {
    "log_analyzer": "./app/agent/graph.py:graph"
  },
  "env": ".env",
  "config": {
    "max_concurrency": 10,
    "timeout": 300,
    "memory_limit": "2GB",
    "cpu_limit": "1.0"
  }
}
```

## Cost Considerations

LangGraph Cloud pricing is based on:
- **Compute time**: Per second of execution
- **Memory usage**: Per GB-second
- **Storage**: For checkpoints and results
- **Network**: For data transfer

Typical costs for this application:
- ~$0.001 per analysis (small logs)
- ~$0.01 per analysis (large logs)

## Advanced Features

### 1. Checkpointing

Enable checkpointing for resumable analyses:

```python
from langgraph.checkpoint.postgres import PostgresSaver

# In your graph creation
checkpointer = PostgresSaver.from_conn_string(
    os.environ["DATABASE_URL"]
)
graph = workflow.compile(checkpointer=checkpointer)
```

### 2. Human-in-the-Loop

Add approval steps:

```python
workflow.add_node("human_approval", human_approval_node)
workflow.add_edge("validate", "human_approval")
```

### 3. Scheduled Runs

Set up periodic log analysis:

```bash
langgraph schedule create \
  --deployment log-analyzer-api \
  --cron "0 */6 * * *" \
  --input '{"log_source": "s3://bucket/logs/latest.log"}'
```

## Troubleshooting

### Deployment Fails

1. Check `langgraph.json` syntax
2. Ensure all dependencies are in `requirements.txt`
3. Verify Python version compatibility

### Runtime Errors

1. Check environment variables are set
2. View logs: `langgraph logs --deployment log-analyzer-api`
3. Check LangSmith traces for detailed errors

### Performance Issues

1. Enable caching in graph nodes
2. Increase memory/CPU limits
3. Use streaming for large logs

## Migration from Self-Hosted

To migrate from Cloud Run to LangGraph Cloud:

1. **Update configuration**: Ensure `langgraph.json` is correct
2. **Test locally**: `langgraph dev`
3. **Deploy**: `langgraph deploy`
4. **Update clients**: Point to new LangGraph Cloud endpoints
5. **Monitor**: Check performance and costs

## Support

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Cloud Docs](https://docs.langchain.com/docs/langgraph-cloud)
- [Community Discord](https://discord.gg/langchain)