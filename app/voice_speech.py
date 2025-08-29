from __future__ import annotations

import os
import queue
import threading
from typing import Callable, Optional, List, Dict

import pyttsx3
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from app.config import voice_cfg


def list_input_devices() -> List[Dict]:
    devices = sd.query_devices()
    inputs: List[Dict] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append(
                {
                    "index": idx,
                    "name": d.get("name", f"Device {idx}"),
                    "default_samplerate": d.get("default_samplerate", None),
                }
            )
    return inputs


class TextToSpeech:
    def __init__(self) -> None:
        pass  # No need to initialize pyttsx3

    def say(self, text: str) -> None:
        try:
            from gtts import gTTS
            from playsound import playsound
            import tempfile, os
            tts = gTTS(text=text, lang='en')
            fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = fp.name
            fp.close()  # Close so gTTS and playsound can access
            tts.save(temp_path)
            playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print(f"gTTS failed: {e}")


class SpeechRecognizer:
    def __init__(self, device_index: Optional[int] = None, debug: bool = False) -> None:
        model_path = voice_cfg.vosk_model_dir
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Vosk model not found at {model_path}. Run the workflow step to download it."
            )
        self.model = Model(model_path)
        self.debug = debug

        # Select device and samplerate
        self.device = device_index
        if device_index is not None:
            dev_info = sd.query_devices(device_index, "input")
        else:
            dev_info = sd.query_devices(sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None, "input")
        samplerate = int(dev_info["default_samplerate"]) if dev_info and dev_info.get("default_samplerate") else voice_cfg.sample_rate
        self.samplerate = samplerate
        self.recognizer = KaldiRecognizer(self.model, self.samplerate)

    def listen_once(self, seconds: float = 5.0) -> str:
        try:
            # Ensure we're using the right format for Vosk
            audio = sd.rec(int(seconds * self.samplerate), samplerate=self.samplerate, channels=1, dtype="int16", device=self.device)
            sd.wait()
            
            # Convert to the exact format Vosk expects
            audio_bytes = audio.tobytes()
            
            # Process in smaller chunks to avoid Vosk errors
            chunk_size = 8000  # 0.5 seconds at 16kHz
            result_text = ""
            
            for i in range(0, len(audio_bytes), chunk_size * 2):  # *2 because int16 = 2 bytes
                chunk = audio_bytes[i:i + chunk_size * 2]
                if len(chunk) > 0:
                    try:
                        if self.recognizer.AcceptWaveform(chunk):
                            import json
                            result = json.loads(self.recognizer.Result())
                            result_text = result.get("text", "")
                            break
                    except Exception as e:
                        if self.debug:
                            print(f"Vosk chunk error: {e}")
                        continue
            
            return result_text.strip()
        except Exception as e:
            if self.debug:
                print(f"Audio recording error: {e}")
            return ""

    def stream_until(self, condition: Callable[[str], bool]) -> str:
        audio_queue: "queue.Queue[bytes]" = queue.Queue()
        transcription = []
        stop_event = threading.Event()

        def audio_callback(indata, frames, time, status):
            if status:
                if self.debug:
                    print(f"Audio callback status: {status}")
            audio_queue.put(bytes(indata))

        def consume():
            partial_accum = ""
            while not stop_event.is_set():
                try:
                    data = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    if self.recognizer.AcceptWaveform(data):
                        import json

                        text = (json.loads(self.recognizer.Result()).get("text", "") or "").strip()
                        if text:
                            transcription.append(text)
                            partial_accum = ""
                            joined = " ".join(transcription)
                            if condition(joined):
                                stop_event.set()
                                break
                    else:
                        # Check partial for wake phrase too
                        import json

                        partial = (json.loads(self.recognizer.PartialResult()).get("partial", "") or "").strip()
                        if partial:
                            partial_accum = partial
                            if self.debug:
                                print(f"[partial] {partial}")
                            combo = (" ".join(transcription + [partial_accum])).strip()
                            if condition(combo):
                                stop_event.set()
                                break
                except Exception as e:
                    if self.debug:
                        print(f"Vosk processing error: {e}")
                    continue

        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
            device=self.device,
        ):
            consumer = threading.Thread(target=consume, daemon=True)
            consumer.start()
            # Wait until condition met
            while not stop_event.is_set():
                stop_event.wait(0.1)
        return " ".join(transcription).strip()

