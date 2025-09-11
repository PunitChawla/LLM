## College Placement QA Chatbot (Local Training + Retrieval)

This project builds a local question-answering chatbot tailored to your college placement data in `placement_qa_dataset_large.csv`. It trains a sentence-embedding model on your dataset and creates a semantic search index to handle paraphrased queries and fuzzy matches, using additional info/tags for context.

### Quickstart (Windows PowerShell)

1. Create virtual environment and install deps:

   - `python -m venv .venv`
   - `./.venv/Scripts/Activate.ps1`
   - `pip install -r requirements.txt`

2. Run the workflow to train and index, then start chat CLI:
   - `powershell -ExecutionPolicy Bypass -File .\workflow.ps1 -Train -BuildIndex`
   - `powershell -ExecutionPolicy Bypass -File .\workflow.ps1 -Chat`

### Files

- `placement_qa_dataset_large.csv`: Source dataset (Q/A, categories, tags)
- `app/config.py`: Paths and hyperparameters
- `app/data_utils.py`: CSV loading and text preparation
- `app/train_embeddings.py`: Fine-tunes a sentence-transformer on your Q/A data
- `app/build_index.py`: Encodes all items and builds a fast cosine-similarity index
- `app/retriever.py`: Loads model + index and performs search
- `app/chat.py`: Interactive CLI that answers queries using the best match and tags
- `app/voice_speech.py`: Speech-to-text (Google Cloud API) and text-to-speech (gTTS)
- `app/voice_wake_and_chat.py`: Direct voice Q&A loop (no wake phrases needed)
- `workflow.ps1`: Orchestrates train/index/chat and voice setup/chat

### Voice Mode

**Setup:**
1. Set your Google Cloud Speech API key in `.env` file:
   ```
   GOOGLE_CLOUD_API_KEY=your_api_key_here
   ```
2. Start voice chat:
   - `powershell -ExecutionPolicy Bypass -File .\workflow.ps1 -VoiceChat`

**Usage:** Simply speak your question directly - no wake phrases needed! The bot will listen, transcribe your question, and speak the answer back. Typing mode remains available with the standard chat command.

### Notes

- Training defaults to 1 epoch for speed. Increase in `app/config.py` if desired.
- The chatbot returns the best-matching answer and includes additional info/tags.
- No external APIs required; everything runs locally.
  Microsoft.QuickAction.WiFi
