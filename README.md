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
- `app/voice_speech.py`: Speech-to-text (Vosk) and text-to-speech (pyttsx3)
- `app/voice_wake_and_chat.py`: Voice wake phrase + voice Q&A loop
- `workflow.ps1`: Orchestrates train/index/chat and voice setup/chat

### Voice Mode

- Setup Vosk model (one-time):
  - `powershell -ExecutionPolicy Bypass -File .\workflow.ps1 -VoiceSetup`
- Start voice chat:
  - `powershell -ExecutionPolicy Bypass -File .\workflow.ps1 -VoiceChat`

Wake phrase: say `hello wake up`. The bot will greet you and listen for your question, then speak the answer. Typing mode remains available with the standard chat command.

### Notes

- Training defaults to 1 epoch for speed. Increase in `app/config.py` if desired.
- The chatbot returns the best-matching answer and includes additional info/tags.
- No external APIs required; everything runs locally.
  Microsoft.QuickAction.WiFi
