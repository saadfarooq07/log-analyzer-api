# LangGraph Cloud SaaS Deployment Guide

This guide shows how to deploy the Log Analyzer API to LangGraph Cloud SaaS - the fully managed deployment option.

## Prerequisites

1. **LangSmith Account**: Sign up at [smith.langchain.com](https://smith.langchain.com)
2. **GitHub Repository**: Your code must be in a GitHub repository
3. **API Keys**: Have your API keys ready:
   - GEMINI_API_KEY
   - GROQ_API_KEY
   - TAVILY_API_KEY

## Step 1: Prepare Your Repository

1. **Push to GitHub** (if not already done):
   ```bash
   cd log-analyzer-api
   git remote add origin https://github.com/YOUR_USERNAME/log-analyzer-api.git
   git push -u origin main
   ```

2. **Verify Required Files**:
   - ✅ `langgraph.json` - Configuration file
   - ✅ `requirements.txt` - Python dependencies
   - ✅ `app/agent/graph.py` - Graph definition with exported `graph` variable

## Step 2: Connect to LangGraph Cloud

1. **Go to LangSmith Console**:
   - Navigate to [smith.langchain.com](https://smith.langchain.com)
   - Sign in with your account

2. **Access LangGraph Deployments**:
   - Click on "LangGraph" in the left sidebar
   - Click "New Deployment"

## Step 3: Create Deployment

1. **Connect GitHub**:
   - Click "Connect GitHub"
   - Authorize LangChain to access your repositories
   - Select your `log-analyzer-api` repository

2. **Configure Deployment**:
   - **Name**: `log-analyzer-api`
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `.` (repository root)
   - **Config File**: `langgraph.json` (auto-detected)

3. **Set Environment Variables**:
   Click "Add Environment Variable" for each:
   - `GEMINI_API_KEY` = your-gemini-api-key
   - `GROQ_API_KEY` = your-groq-api-key
   - `TAVILY_API_KEY` = your-tavily-api-key

4. **Deploy**:
   - Click "Create Deployment"
   - LangGraph will automatically:
     - Build your application
     - Run tests (if any)
     - Deploy to cloud infrastructure

## Step 4: Access Your Deployment

Once deployed, you'll get:

1. **API Endpoint**:
   ```
   https://YOUR_DEPLOYMENT_ID.api.langchain.com
   ```

2. **API Key**:
   - Found in your LangSmith settings
   - Required for API authentication

## Step 5: Use the Deployed API

### Using LangGraph SDK

```python
from langgraph_sdk import get_client

# Initialize client
client = get_client(
    url="https://YOUR_DEPLOYMENT_ID.api.langchain.com",
    api_key="your-langsmith-api-key"
)

# Create a thread
thread = await client.threads.create()

# Run analysis
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="log_analyzer",
    input={
        "log_content": "ERROR: Database connection failed",
        "environment_details": {"os": "Linux"},
        "application_name": "web-api",
        "analysis_type": "general",
        "include_suggestions": True,
        "include_documentation": True,
        "messages": [],
        "current_analysis": None,
        "validation_result": None,
        "tool_calls": [],
        "analysis_result": None,
        "error": None
    }
)

# Wait for completion
await client.runs.wait(thread["thread_id"], run["run_id"])

# Get results
final_state = await client.threads.get_state(thread["thread_id"])
print(final_state["values"]["analysis_result"])
```

### Using REST API

```bash
# Create a thread
curl -X POST https://YOUR_DEPLOYMENT_ID.api.langchain.com/threads \
  -H "x-api-key: your-langsmith-api-key" \
  -H "Content-Type: application/json"

# Response: {"thread_id": "abc123..."}

# Run analysis
curl -X POST https://YOUR_DEPLOYMENT_ID.api.langchain.com/threads/abc123/runs \
  -H "x-api-key: your-langsmith-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "log_analyzer",
    "input": {
      "log_content": "ERROR: Database connection failed",
      "environment_details": {"os": "Linux"},
      "application_name": "web-api",
      "analysis_type": "general",
      "include_suggestions": true,
      "include_documentation": true,
      "messages": [],
      "current_analysis": null,
      "validation_result": null,
      "tool_calls": [],
      "analysis_result": null,
      "error": null
    }
  }'

# Get state
curl https://YOUR_DEPLOYMENT_ID.api.langchain.com/threads/abc123/state \
  -H "x-api-key: your-langsmith-api-key"
```

### Streaming Results

```python
# Stream events
async for event in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="log_analyzer",
    input={...},
    stream_mode="events"
):
    print(f"Event: {event['event']}")
    if event['event'] == 'on_chain_end' and event['name'] == 'analyze':
        print("Analysis complete!")
```

## Step 6: Monitor Your Deployment

### View Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project
3. View:
   - Execution traces
   - Latency metrics
   - Error logs
   - Token usage

### Check Deployment Status

In the LangGraph deployments page, you can:
- View deployment status
- Check build logs
- Monitor usage metrics
- Update environment variables
- Redeploy from new commits

## Continuous Deployment

LangGraph Cloud automatically redeploys when you push to your connected branch:

1. Make changes locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Update analysis logic"
   git push origin main
   ```
3. LangGraph Cloud automatically:
   - Detects the change
   - Rebuilds your application
   - Deploys the new version
   - Maintains zero downtime

## Advanced Features

### 1. Human-in-the-Loop

Add human approval to your graph:

```python
# In your graph definition
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["validate"])
```

### 2. Scheduled Runs

Set up periodic analysis (coming soon in Cloud SaaS).

### 3. Webhooks

Configure webhooks for run completion:

```python
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="log_analyzer",
    input={...},
    webhook="https://your-server.com/webhook"
)
```

## Troubleshooting

### Build Failures

1. Check build logs in the deployment page
2. Common issues:
   - Missing dependencies in `requirements.txt`
   - Import errors in graph definition
   - Environment variable not set

### Runtime Errors

1. View traces in LangSmith
2. Check for:
   - API key errors
   - Model quota exceeded
   - Timeout issues

### Performance Issues

1. Monitor latency in LangSmith
2. Optimize by:
   - Reducing prompt size
   - Caching results
   - Using streaming for large logs

## Pricing

LangGraph Cloud SaaS pricing:
- Based on number of node executions
- Includes infrastructure and management
- See [pricing page](https://www.langchain.com/pricing-langgraph-platform)

## Support

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Support](https://smith.langchain.com/support)
- [Discord Community](https://discord.gg/langchain)