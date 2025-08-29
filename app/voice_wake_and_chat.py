from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from app.config import voice_cfg
from app.retriever import Retriever
from app.voice_speech import SpeechRecognizer, TextToSpeech, list_input_devices


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def voice_chat(device_index: int | None = None, debug: bool = False) -> None:
    console = Console()
    tts = TextToSpeech()
    asr = SpeechRecognizer(device_index=device_index, debug=debug)
    retriever = Retriever()

    phrases = [normalize(p) for p in voice_cfg.wake_phrases]
    console.print(Panel(f"Say a wake phrase: {', '.join(voice_cfg.wake_phrases)}", title="Voice Mode"))

    import time
    while True:
        # Wait for wake phrase
        heard = asr.stream_until(lambda text: any(p in normalize(text) for p in phrases))
        if any(p in normalize(heard) for p in phrases):
            greeting = "Hello, my name is Arya Chatbot. How can I help you?"
            console.print(Panel(greeting, title="Wake"))
            tts.say(greeting)

            # Stay in active listening mode for 5 minutes or until exit
            active_start = time.time()
            while time.time() - active_start < 300:  # 5 minutes
                console.print("Listening for your question... (say 'exit' to quit)")
                question = asr.listen_once(seconds=5.0)
                norm_q = normalize(question)
                if not norm_q:
                    tts.say("I didn't hear a question. Please ask again or say 'exit' to quit.")
                    continue
                if norm_q in {"exit", "quit", "bye"}:
                    tts.say("Goodbye!")
                    return

                # Get answer and respond quickly
                results = retriever.search(question, top_k=3)
                if not results:
                    answer = "I could not find an answer."
                else:
                    _, hit = results[0]
                    answer = str(hit.get("answers", "")).strip() or "I could not find an answer."

                console.print(Panel(f"You: {question}\nAnswer: {answer}", title="Response"))
                tts.say(answer)
            tts.say("Wake phase ended. Say the wake phrase to activate again.")


if __name__ == "__main__":
    voice_chat()

