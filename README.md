# SelfRAG

A self-evaluating Retrieval-Augmented Generation (RAG) system that leverages OpenAI's GPT-4 and FAISS for context-aware question answering with built-in response evaluation and refinement.

## Features

- Document embedding and similarity search using FAISS
- Context-based response generation with GPT-4
- Automatic response evaluation and refinement
- Configurable context retrieval (k-nearest neighbors)

## Installation

```bash
pip install openai faiss-cpu numpy python-dotenv
```

## Environment Setup

Create a `.env` file with your OpenAI API key:
```
OPENAI_KEY=your_api_key_here
```

## Usage

```python
from selfrag import SelfRAG

# Initialize
rag = SelfRAG(api_key="your_api_key")

# Add documents
docs = [
    "Document 1 content",
    "Document 2 content"
]
rag.add_documents(docs)

# Query
result = rag.query("Your question here")
print(result["final_response"])
```

## Response Structure

The query method returns a dictionary containing:
- `context`: Retrieved relevant documents
- `initial_response`: First generated response
- `evaluation`: Assessment scores and improvement suggestions
- `final_response`: Refined answer based on evaluation

## System Components

- `add_documents()`: Embeds and indexes documents using FAISS
- `generate_response()`: Creates initial response using retrieved context
- `evaluate_response()`: Rates response quality on relevance, context usage, and accuracy
- `refine_response()`: Improves response based on evaluation feedback
- `query()`: Orchestrates the entire RAG pipeline

## Dependencies

- OpenAI
- FAISS
- NumPy
- python-dotenv