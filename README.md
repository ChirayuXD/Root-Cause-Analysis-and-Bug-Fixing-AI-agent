# Root Cause Analysis (RCA) & Bug Fixing AI Agent

This project is an automated AI-driven pipeline designed to streamline the software debugging process. It integrates Jira and GitHub to fetch issue details, performs Root Cause Analysis (RCA) using LLMs, retrieves relevant code files, indexes them for semantic search, and generates actionable code fixes.

The entire workflow is orchestrated via a user-friendly Gradio interface.

## 🚀 Features

* **Jira Integration**: Automatically fetches recent issues and analyzes attached logs to identify error patterns.
* **AI-Powered RCA**: Uses CrewAI agents (powered by Azure GPT-4o) to generate detailed Root Cause Analysis reports based on logs and issue summaries.
* **Smart Code Retrieval**: Scans the provided GitHub repository to download specific files identified in the RCA.
* **Semantic Code Search**: Chunks and indexes code using FAISS and Sentence Transformers to allow the AI to find the exact context of bugs.
* **Automated Code Fixing**: A specialized AI agent iteratively analyzes code chunks and generates unified diffs/patches to fix the bugs.
* **Interactive UI**: A web-based dashboard built with Gradio to trigger pipelines and view reports in real-time.

## 🛠️ Architecture Pipeline

1. **Jira Issue & Log Analysis**: Fetches ticket info and extracts error stacks from log attachments.
2. **RCA Generation**: An AI Agent ("Root Cause Analysis Expert") acts on the log data to hypothesize the root cause and identify culprit files.
3. **GitHub File Fetcher**: Downloads the specific source code files mentioned in the RCA from the repository.
4. **FAISS Indexing**: Parses code (supports Java structure) and builds a vector database for semantic search.
5. **Code Fix Generation**: An AI Agent ("Code Chunk Analyzer") queries the vector DB to understand the code context and writes the fix.

## 📋 Prerequisites

* **Python 3.10+**
* **Jira Account**: Access to a Jira instance with a Personal Access Token.
* **GitHub Account**: A Personal Access Token (Classic) with repo read permissions.
* **Azure OpenAI**: Access to Azure OpenAI Service (specifically gpt-4o).

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Create a Virtual Environment** (Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## 🔐 Configuration

You must create a `.env` file in the root directory to store your credentials. The application will not run without these.

**Create a file named `.env`:**
```ini
# Jira Configuration
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_BEARER_TOKEN=your_jira_pat_token

# GitHub Configuration
GITHUB_TOKEN=your_github_pat_token

# Azure OpenAI / CrewAI Configuration
# The agents are configured to use "azure/gpt-4o"
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

## ▶️ Usage

### Running Locally

Execute the main script to launch the Gradio interface:
```bash
python main.py
```

Once running, open your browser and navigate to: `http://localhost:7860`

**In the Interface:**

1. Enter the **Jira Project ID** (e.g., `PROJ123`).
2. Set **Days Back** (how far back to look for issues).
3. Enter the **GitHub Repository URL** (e.g., `https://github.com/org/repo`).
4. Click **Run Pipeline**.

### Running with Docker

1. **Build the image:**
```bash
docker build -t rca-agent .
```

2. **Run the container:**  
   Note: You must pass your environment variables to the container.
```bash
docker run -p 7860:7860 --env-file .env rca-agent
```

## 📂 Project Structure

* `main.py`: Entry point. Runs the Gradio UI and orchestrates the subprocess pipeline.
* `JIRA_issue_log_analysis.py`: Handles connection to Jira and extraction of error logs.
* `rca_bot.py`: CrewAI agent that generates the text-based RCA report.
* `github_fetcher.py`: Logic to search and download files from GitHub based on RCA output.
* `faiss_code_file_indexer.py`: Parses code (Java/Text), creates embeddings, and builds the FAISS index.
* `code_fix_ai.py`: CrewAI agent that queries the FAISS index and generates the code fixes.
* `Dockerfile`: Configuration for containerizing the application.

## ⚠️ Notes

* **Java Parsing**: The `faiss_code_file_indexer.py` has specific logic for parsing Java files (methods/classes). If `javalang` is missing, it falls back to text chunking.
* **LLM Cost**: This tool uses GPT-4o via Azure. Be aware of API costs associated with processing large logs or codebases.
