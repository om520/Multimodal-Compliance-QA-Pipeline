# Azure Cloud Compliance QA Pipeline

This directory contains the original **Azure Cloud Architecture** for the Multimodal Compliance QA Pipeline before the local open-source model migration.

---

## 1. Cloud Architecture Overview

The cloud pipeline relies on three managed Azure services:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AZURE CLOUD PIPELINE                              │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│ Azure Video Indexer     │ Azure AI Search         │ Azure OpenAI            │
│ (AVAM / ARM API)        │ (Vector & Hybrid Store) │ (GPT-4o / GPT-4 Turbo)  │
│ • Speech Transcription  │ • HNSW Vector Profile   │ • Compliance Violation  │
│ • OCR Frame Extraction  │ • Semantic Ranker       │   Auditor               │
│ • Visual Scene Insights │ • Document Chunks       │ • Executive Reporter    │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 2. File Manifest

* **`azure_nodes.py`**: The LangGraph nodes calling Azure services (`AzureChatOpenAI`, `AzureOpenAIEmbeddings`, `SearchClient`).
* **`azure_video_indexer.py`**: Client for Azure Video Indexer (generates ARM access tokens, uploads videos, polls indexing status, and parses insights).
* **`azure_index_documents.py`**: Indexes compliance PDFs into Azure AI Search with vector search profiles and semantic configurations.
* **`azure_workflow.py`**: LangGraph graph topology compiled for Azure execution.
* **`azure_state.py`**: Shared state schema (`VideoAudioState`, `ComplianceIssue`).
* **`azure_server.py`**: FastAPI REST API server wired to Azure LangGraph workflow.
* **`azure_telemetry.py`**: Azure Application Insights and OpenTelemetry exporters.
* **`azure.env.example`**: Environment variable template required for Azure authentication.

---

## 3. Required Azure Environment Variables

Copy `azure.env.example` to `.env` and populate:

```ini
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-openai-key>
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_AI_SEARCH_KEY=<your-search-admin-key>
AZURE_AI_SEARCH_INDEX_NAME=compliance-rules-index

# Azure Video Indexer
AZURE_VIDEO_INDEXER_ACCOUNT_ID=<your-vi-account-id>
AZURE_VIDEO_INDEXER_LOCATION=trial
AZURE_VIDEO_INDEXER_API_KEY=<your-vi-key>
```

