from __future__ import annotations

import argparse
import json
from typing import Optional

from app.voice_speech import list_input_devices
from app.voice_wake_and_chat import voice_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice utilities for Arya Chatbot")
    parser.add_argument("--list-audio", action="store_true", help="List input audio devices as JSON")
    parser.add_argument("--voice-chat", action="store_true", help="Start voice chat with wake phrase")
    parser.add_argument("--device", type=int, default=None, help="Input device index to use")
    parser.add_argument("--debug", action="store_true", help="Print partial ASR results while listening")
    args = parser.parse_args()

    if args.list_audio:
        print(json.dumps(list_input_devices(), indent=2))
        return

    if args.voice_chat:
        voice_chat(device_index=args.device, debug=args.debug)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


