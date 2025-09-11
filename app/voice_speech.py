"""
Clean voice speech module using only Google Cloud Speech API
"""
from __future__ import annotations

import os
import tempfile
import requests
import json
import base64
import time
import wave
import pyaudio
import re
from typing import Optional

from app.config import voice_cfg


def list_input_devices() -> list[dict]:
    """List available audio input devices using PyAudio."""
    audio = pyaudio.PyAudio()
    devices = []
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Input device
            devices.append({
                'index': i,
                'name': device_info['name'],
                'channels': device_info['maxInputChannels']
            })
    
    audio.terminate()
    return devices


class TextToSpeech:
    """Text-to-Speech using Google Text-to-Speech (gTTS)"""
    
    def __init__(self) -> None:
        pass

    def say(self, text: str) -> None:
        """Convert text to speech and play it."""
        try:
            from gtts import gTTS
            from playsound import playsound
            
            # Create TTS audio
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
            
            tts.save(temp_path)
            playsound(temp_path)
            os.remove(temp_path)
            
        except Exception as e:
            print(f"TTS failed: {e}")


class SpeechRecognizer:
    """Speech Recognition using Google Cloud Speech-to-Text API"""
    
    def __init__(self, device_index: Optional[int] = None, debug: bool = False, api_key: Optional[str] = None) -> None:
        self.debug = debug
        self.device = device_index
        
        # Audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Get API key
        self.api_key = api_key or voice_cfg.google_cloud_api_key
        if not self.api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY not found in environment variables")
        
        # Google Cloud Speech API endpoint
        self.api_url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"
        
        if self.debug:
            print(f"✅ Using Google Cloud Speech API with PyAudio")

    def listen_once(self, seconds: float = 5.0) -> str:
        """Record audio and transcribe using Google Cloud Speech API."""
        if self.debug:
            print(f"🎤 Recording for {seconds} seconds...")
        
        try:
            # Record audio using PyAudio
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.device,
                frames_per_buffer=self.CHUNK
            )
            
            frames = []
            for i in range(0, int(self.RATE / self.CHUNK * seconds)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Save to temporary file with better handling
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()  # Close the file handle before writing
            
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            # Transcribe using Google Cloud API
            transcript = self._transcribe_with_api(temp_path)
            
            if self.debug:
                print(f"🔍 Got transcript from API: '{transcript}'")
            
            # Clean up temporary file
            self._cleanup_temp_file(temp_path)
            
            if self.debug:
                print(f"🔍 Final transcript from listen_once: '{transcript}'")
            return transcript
            
        except Exception as e:
            if self.debug:
                print(f"❌ STT recording error: {e}")
            return ""

    def _transcribe_with_api(self, audio_file_path: str) -> str:
        """Transcribe audio using Google Cloud Speech API."""
        if self.debug:
            print("🔄 Processing audio...")
        
        transcript_result = ""
        
        try:
            # Read and encode audio file
            with open(audio_file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": self.RATE,
                    "languageCode": "en-US",  # Use US English only
                    "enableAutomaticPunctuation": True,
                    "enableWordTimeOffsets": False,
                    "enableWordConfidence": True,
                    "model": "latest_long",
                    "useEnhanced": True,
                    "maxAlternatives": 3,
                    # Add speech contexts for better recognition of specific terms
                    "speechContexts": [{
                        "phrases": [
                            "Mohit Mishra",
                            "Arun Arya",
                            "Dr. Arun Arya",
                            "Prof. Arun Arya",
                            "Professor Arun Arya",
                            "faculty",
                            "professor", 
                            "sir",
                            "ma'am",
                            "teacher",
                            "department",
                            "Mechanical Engineering",
                            "Computer Science Engineering",
                            "Electrical Engineering",
                            "Civil Engineering",
                            "Electronics and Communication Engineering",
                            "Information Technology",
                            "CSE",
                            "ECE",
                            "IT",
                            "HOD",
                            "Head of Department",
                            "Arya College",
                            "placement",
                            "admission"
                        ],
                        "boost": 20
                    }]
                },
                "audio": {
                    "content": audio_base64
                }
            }
            
            # Make API request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse results with multiple alternatives
                if 'results' in result and result['results']:
                    best_transcript = ""
                    best_confidence = 0.0
                    all_alternatives = []
                    
                    for result_item in result['results']:
                        alternatives = result_item.get('alternatives', [])
                        for alternative in alternatives:
                            if 'transcript' in alternative and alternative['transcript'].strip():
                                transcript = alternative['transcript'].strip()
                                confidence = alternative.get('confidence', 0.0)
                                all_alternatives.append((transcript, confidence))
                                
                                if confidence > best_confidence:
                                    best_transcript = transcript
                                    best_confidence = confidence
                    
                    if self.debug and all_alternatives:
                        print(f"🔍 All alternatives:")
                        for i, (trans, conf) in enumerate(all_alternatives):
                            print(f"   {i+1}. {trans} (confidence: {conf:.2f})")
                    
                    if best_transcript:
                        # Filter out non-English text
                        if self._is_english_text(best_transcript):
                            # Apply name correction
                            corrected_transcript = self._correct_faculty_names(best_transcript)
                            if self.debug:
                                if corrected_transcript != best_transcript:
                                    print(f"📝 Original: {best_transcript}")
                                    print(f"✨ Corrected: {corrected_transcript} (confidence: {best_confidence:.2f})")
                                else:
                                    print(f"📝 Transcript: {corrected_transcript} (confidence: {best_confidence:.2f})")
                            return corrected_transcript
                        else:
                            if self.debug:
                                print(f"❌ Non-English text detected, skipping: {best_transcript}")
                            return ""
                
                # No valid transcript found
                if self.debug:
                    print("❌ No clear speech detected")
                return ""
            else:
                if self.debug:
                    print(f"❌ API Error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            if self.debug:
                print(f"❌ Transcription error: {str(e)}")
            return transcript_result

    def _correct_faculty_names(self, transcript: str) -> str:
        """Correct common misrecognitions of faculty names and terms."""
        corrections = {
            # Faculty names
            "mohit mishra": "Mohit Mishra",
            "mohit misra": "Mohit Mishra", 
            "mohit meshes": "Mohit Mishra",
            "mohit mission": "Mohit Mishra",
            "mohit misha": "Mohit Mishra",
            "mohit sharma": "Mohit Mishra",
            "mode mishra": "Mohit Mishra",
            "mobile mishra": "Mohit Mishra",
            "mohit mitra": "Mohit Mishra",
            "mohit mehra": "Mohit Mishra",
            
            # Arun Arya variations
            "arun arya": "Arun Arya",
            "arun aria": "Arun Arya", 
            "aaron arya": "Arun Arya",
            "run arya": "Arun Arya",
            "dr. arun arya": "Dr. Arun Arya",
            "doctor arun arya": "Dr. Arun Arya",
            "prof arun arya": "Prof. Arun Arya",
            "professor arun arya": "Prof. Arun Arya",
            
            # Department names
            "mechanical department": "Mechanical Department",
            "mechanic department": "Mechanical Department",  
            "mechanical departement": "Mechanical Department",
            "mechanical engineering": "Mechanical Engineering",
            "computer science department": "Computer Science Department",
            "computer department": "Computer Science Department",
            "computer science engineering": "Computer Science Engineering",
            "cse": "Computer Science Engineering",
            "electrical department": "Electrical Department",
            "electrical engineering": "Electrical Engineering",
            "civil department": "Civil Department",
            "civil engineering": "Civil Engineering",
            "electronics and communication": "Electronics and Communication Engineering",
            "ece": "Electronics and Communication Engineering",
            "information technology": "Information Technology",
            "it department": "Information Technology",
            
            # Common terms
            "hod": "HOD",
            "head of department": "Head of Department",
            "sir": "Sir",
            "professor": "Professor",
            "faculty": "Faculty",
            "teacher": "Teacher",
            "maam": "Ma'am",
            "madam": "Ma'am",
        }
        
        corrected = transcript
        lower_transcript = transcript.lower()
        
        # Apply corrections
        for wrong, correct in corrections.items():
            if wrong in lower_transcript:
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                corrected = pattern.sub(correct, corrected)
        
        return corrected

    def _is_english_text(self, text: str) -> bool:
        """Check if the text is primarily in English characters."""
        if not text or not text.strip():
            return False
        
        # Count English characters (letters, numbers, common punctuation)
        english_chars = sum(1 for c in text if c.isascii() and (c.isalnum() or c in ' .,?!-\'\"()'))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        # Consider it English if at least 80% are ASCII characters
        english_ratio = english_chars / total_chars
        return english_ratio >= 0.8

    def _cleanup_temp_file(self, temp_path: str) -> None:
        """Clean up temporary file with retry logic."""
        try:
            os.unlink(temp_path)
            if self.debug:
                print("✅ Temp file cleaned up successfully")
        except OSError as e:
            if self.debug:
                print(f"⚠️ Could not delete temp file immediately: {e}")
            # Try again after a short delay
            time.sleep(0.1)
            try:
                os.unlink(temp_path)
                if self.debug:
                    print("✅ Temp file cleaned up after delay")
            except OSError:
                if self.debug:
                    print("⚠️ Temp file cleanup failed, but continuing...")
