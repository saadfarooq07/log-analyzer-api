"""Setup configuration for log-analyzer-api package."""

from setuptools import setup, find_packages

setup(
    name="log-analyzer-api",
    version="1.0.0",
    description="Log Analyzer API using LangGraph and AI models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "gunicorn>=21.2.0",
        "langgraph>=0.0.32",
        "langchain>=0.1.0",
        "langchain-google-genai>=0.0.5",
        "langchain-groq>=0.0.3",
        "langchain-community>=0.0.10",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.2",
        "sse-starlette>=1.8.2",
        "tavily-python>=0.3.0",
    ],
    entry_points={
        "console_scripts": [
            "log-analyzer-api=app.main:main",
        ],
    },
)