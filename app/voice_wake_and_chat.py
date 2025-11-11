from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from app.config import voice_cfg
from app.retriever import Retriever
from app.voice_speech import SpeechRecognizer, TextToSpeech, list_input_devices


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def correct_text(text: str) -> str:
    corrections = {
        "arya college": "Arya College",
        "mechanical development": "mechanical department",
        # Add more custom corrections as needed
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def voice_chat(device_index: int | None = None, debug: bool = False, google_api_key: str | None = None) -> None:
    console = Console()
    tts = TextToSpeech()
    asr = SpeechRecognizer(device_index=device_index, debug=debug, api_key=google_api_key)
    retriever = Retriever()

    greeting = "Hello, my name is Arya Chatbot. I'm ready to answer your questions!"
    console.print(Panel(greeting, title="Voice Chat Ready"))
    tts.say(greeting, asr)  # Pass ASR for interrupt capability

    import time
    while True:
        console.print("Listening for your question... (say 'exit' to quit)")
        # Process audio in chunks for faster response  
        transcript = ""
        for i in range(2):  # Try up to 2 chunks (2x5s = 10s max)
            if debug:
                print(f"[Debug] Listening for audio chunk {i+1}/2...")
            part = asr.listen_once(seconds=5.0)
            if debug:
                print(f"[Debug] Audio chunk {i+1} result: '{part}'")
            if part:
                transcript += " " + part
                # If we have a reasonable question, break early
                if len(transcript.strip()) > 8:
                    break
        
        if debug:
            print(f"[Debug] Full transcript: '{transcript.strip()}'")
        question = correct_text(transcript.strip())
        if debug:
            print(f"[Debug] Corrected question: '{question}'")
        norm_q = normalize(question)
        if debug:
            print(f"[Debug] Normalized question: '{norm_q}'")
        
        if not norm_q:
            if debug:
                print("[Debug] No valid question detected")
            tts.say("I didn't hear a question. Please ask again or say 'exit' to quit.", asr)
            continue
        if norm_q in {"exit", "quit", "bye"}:
            tts.say("Goodbye!", asr)
            return
        
        # Get answer and respond quickly
        results = retriever.search(question, top_k=3)
        if not results:
            answer = "I could not find an answer."
        else:
            _, hit = results[0]
            answer = str(hit.get("answers", "")).strip() or "I could not find an answer."

        console.print(Panel(f"You: {question}\nAnswer: {answer}", title="Response"))
        console.print("🎧 Speaking... (Press SPACE or ENTER to interrupt)", style="dim")
        tts.say(answer, asr)  # Pass ASR for interrupt capability
        
        # Small delay to separate speech from next listening
        import time
        time.sleep(0.5)


if __name__ == "__main__":
    voice_chat()
