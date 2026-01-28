"""
Streaming ASR with NeMo - FastAPI Server

This module provides:
1. StreamingASR class - processes audio chunks and returns transcription with metadata
2. FastAPI server - WebSocket and HTTP endpoints for real-time speech recognition

Endpoints:
    WebSocket: /listen/{session_id}  - Real-time streaming transcription
    POST:      /transcribe           - Transcribe uploaded audio file
    GET:       /status               - Server status and GPU info
    GET:       /                     - Health check

Production URL: wss://audio.whissle.ai/listen/{session_id}

Usage:
    # Start server:
    python asr_streaming_bufferd.py --port 8000
    
    # Or import and use directly:
    from asr_streaming_bufferd import StreamingASR
    asr = StreamingASR(model_path="path/to/model.nemo")
    result = asr.process_chunk(audio_bytes)

Concurrency:
    - Multiple WebSocket connections are supported
    - Each session maintains isolated audio buffer
    - GPU inference is serialized (one at a time via lock)
    - For higher throughput, run multiple instances behind nginx
"""

import os
import io
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import wave
import asyncio
from contextlib import asynccontextmanager

# ============================================================
# Streaming ASR Class
# ============================================================

@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str                    # Full current window transcription
    text_only: str               # Text without metadata tags
    incremental_text: str        # NEW: Only the new/incremental text
    metadata: Dict[str, str]     # AGE, GENDER, EMOTION, INTENT
    chunk_duration: float
    buffer_duration: float       # Duration of audio in buffer
    is_final: bool = False       # True if this is end-of-stream


class StreamingASR:
    """
    Streaming ASR processor using NeMo CTC model with sliding window.
    
    Key features:
    - Sliding window buffer (keeps last N seconds, not full history)
    - Incremental output (returns only NEW text each chunk)
    - Tracks committed vs in-progress text
    
    Processes audio chunks and returns:
    - Regular text (with inline ENTITY tokens)
    - Metadata tags (AGE, GENDER, EMOTION, INTENT)
    - Incremental text (only what's new since last chunk)
    """
    
    METADATA_PREFIXES = ('AGE_', 'GENDER_', 'EMOTION_', 'INTENT_', 'SPEAKER')
    ENTITY_PREFIXES = ('ENTITY_', 'END')
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        buffer_duration: float = 6.0,   # Sliding window size in seconds
        overlap_duration: float = 1.0,  # Overlap for context
        device: str = None
    ):
        """
        Initialize streaming ASR.
        
        Args:
            model_path: Path to .nemo model file
            sample_rate: Audio sample rate (default 16000)
            buffer_duration: Max audio buffer duration in seconds (default 15)
            overlap_duration: Overlap when trimming buffer (default 2)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self.max_samples = int(buffer_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_path}...")
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(
            model_path, 
            map_location=self.device
        )
        self.model.eval()
        
        # Get vocabulary
        self.vocab = list(self.model._cfg.decoder.vocabulary)
        self.blank_id = len(self.vocab)
        
        # Session state
        self.reset()
        
        print(f"StreamingASR initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Vocab size: {len(self.vocab)}")
        print(f"  - Sample rate: {self.sample_rate}")
        print(f"  - Buffer: {buffer_duration}s, Overlap: {overlap_duration}s")
    
    def reset(self):
        """Reset session state for new audio stream."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.committed_text = ""      # Text that's been "committed" (won't change)
        self.last_window_text = ""    # Last transcription of current window
        self.total_audio_processed = 0.0  # Total seconds processed (including trimmed)
    
    def _ctc_decode(self, predictions: np.ndarray) -> List[int]:
        """CTC greedy decode: collapse duplicates, remove blanks."""
        decoded = []
        prev = self.blank_id
        for p in predictions:
            if p != prev and p != self.blank_id:
                decoded.append(int(p))
            prev = p
        return decoded
    
    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to formatted text."""
        if not token_ids:
            return ''
        
        tokens = []
        for tid in token_ids:
            if tid < len(self.vocab):
                token = self.vocab[tid]
                # Add space before special tokens
                if any(token.startswith(p) for p in self.METADATA_PREFIXES + self.ENTITY_PREFIXES):
                    if tokens and not tokens[-1].endswith(' '):
                        tokens.append(' ')
                tokens.append(token)
        
        text = ''.join(tokens).replace('▁', ' ')
        return text.strip()
    
    def _extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata tags from text."""
        import re
        metadata = {}
        
        for prefix in self.METADATA_PREFIXES:
            pattern = rf'{prefix}(\w+)'
            match = re.search(pattern, text)
            if match:
                key = prefix.rstrip('_').lower()
                metadata[key] = f"{prefix}{match.group(1)}"
        
        return metadata
    
    def _get_text_only(self, text: str) -> str:
        """Remove ALL metadata tags from text, keep only speech + entities."""
        import re
        text_only = text
        # Remove metadata tags (AGE_, GENDER_, EMOTION_, INTENT_, SPEAKER)
        for prefix in self.METADATA_PREFIXES:
            text_only = re.sub(rf'\s*{prefix}\w*', '', text_only)
        # Clean up any leftover partial tags like "18_30" or "30_45"
        text_only = re.sub(r'\s*\d+_\d+\s*', ' ', text_only)
        # Clean up multiple spaces
        text_only = re.sub(r'\s+', ' ', text_only)
        return text_only.strip()
    
    def _get_clean_text(self, text: str) -> str:
        """Remove ALL tags (metadata + entities) from text, return plain speech only."""
        import re
        clean = self._get_text_only(text)
        # Remove entity tags: ENTITY_TYPE ... END
        clean = re.sub(r'ENTITY_\w+\s*', '', clean)
        clean = re.sub(r'\s*END\b', '', clean)
        # Clean up multiple spaces
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities with their types and values from text.
        
        Returns list of dicts: [{"type": "LOCATION", "value": "London"}, ...]
        """
        import re
        entities = []
        
        # Pattern: ENTITY_TYPE value END
        pattern = r'ENTITY_(\w+)\s+(.+?)\s+END'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entity_type = match.group(1)
            entity_value = match.group(2).strip()
            entities.append({
                "type": entity_type,
                "value": entity_value,
                "raw": match.group(0)
            })
        
        return entities
    
    def _find_stable_boundary(self, text: str) -> int:
        """
        Find a stable boundary in text where we can commit.
        Returns index of where to split, or 0 if no good boundary.
        
        Looks for sentence boundaries or natural pauses.
        """
        import re
        
        # Look for sentence-ending punctuation followed by space
        matches = list(re.finditer(r'[.!?]\s+', text))
        if matches:
            # Take the last complete sentence
            return matches[-1].end()
        
        # Look for comma or natural pause points
        matches = list(re.finditer(r',\s+', text))
        if matches and len(text) > 50:
            return matches[-1].end()
        
        return 0
    
    @torch.no_grad()
    def process_chunk(
        self, 
        audio_chunk: np.ndarray,
        is_final: bool = False
    ) -> TranscriptionResult:
        """
        Process an audio chunk with sliding window.
        
        Args:
            audio_chunk: Audio samples (int16 or float32)
            is_final: True if this is the last chunk (forces commit)
            
        Returns:
            TranscriptionResult with incremental text
        """
        # Normalize audio
        if len(audio_chunk) > 0:
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
        
        chunk_duration = len(audio_chunk) / self.sample_rate if len(audio_chunk) > 0 else 0
        
        # Add to buffer
        if len(audio_chunk) > 0:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Check if we need to trim buffer (sliding window)
        trimmed_text = ""
        if len(self.audio_buffer) > self.max_samples:
            # Transcribe before trimming to capture text we're about to lose
            pre_trim_text = self._transcribe_buffer()
            
            # Find a good boundary to commit
            boundary = self._find_stable_boundary(pre_trim_text)
            if boundary > 0:
                trimmed_text = pre_trim_text[:boundary]
                self.committed_text += trimmed_text
            
            # Trim buffer, keeping overlap for context
            trim_amount = len(self.audio_buffer) - self.max_samples + self.overlap_samples
            self.audio_buffer = self.audio_buffer[trim_amount:]
            self.total_audio_processed += trim_amount / self.sample_rate
        
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Transcribe current buffer
        if len(self.audio_buffer) > 0:
            window_text = self._transcribe_buffer()
        else:
            window_text = ""
        
        # Calculate incremental text (what's new)
        # Compare with last window transcription
        incremental = self._get_incremental(self.last_window_text, window_text)
        self.last_window_text = window_text
        
        # If final, commit everything
        if is_final and window_text:
            full_text = self.committed_text + window_text
            incremental = window_text  # Return full window as final
        else:
            full_text = self.committed_text + window_text
        
        # Extract metadata from current window
        metadata = self._extract_metadata(window_text)
        text_only = self._get_text_only(window_text)
        
        return TranscriptionResult(
            text=full_text,
            text_only=self.committed_text + text_only,
            incremental_text=incremental,
            metadata=metadata,
            chunk_duration=chunk_duration,
            buffer_duration=buffer_duration,
            is_final=is_final
        )
    
    def _transcribe_buffer(self) -> str:
        """Transcribe current audio buffer."""
        return self.transcribe_audio(self.audio_buffer)
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Transcribe audio directly (thread-safe - doesn't modify instance state).
        
        Use this for concurrent transcription from multiple sessions.
        The GPU forward pass should still be protected by an external lock.
        """
        if len(audio) == 0:
            return ""
        
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([len(audio)]).to(self.device)
        
        with torch.no_grad():  # Ensure no gradient computation
            log_probs, encoded_len, predictions = self.model.forward(
                input_signal=audio_tensor,
                input_signal_length=audio_len
            )
        
        pred_ids = predictions[0].cpu().numpy()
        token_ids = self._ctc_decode(pred_ids)
        return self._ids_to_text(token_ids)
    
    def _get_incremental(self, old_text: str, new_text: str) -> str:
        """
        Get incremental difference between old and new text.
        Returns only the new portion.
        """
        if not old_text:
            return new_text
        if not new_text:
            return ""
        
        # Find longest common prefix
        min_len = min(len(old_text), len(new_text))
        common_len = 0
        for i in range(min_len):
            if old_text[i] == new_text[i]:
                common_len = i + 1
            else:
                break
        
        # Return new portion
        return new_text[common_len:].lstrip()
    
    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file."""
        import scipy.io.wavfile as wav
        
        sr, audio = wav.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Mono
        
        # Resample if needed
        if sr != self.sample_rate:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, 
                int(len(audio) * self.sample_rate / sr)
            )
        
        self.reset()
        return self.process_chunk(audio, is_final=True)


# ============================================================
# FastAPI Server
# ============================================================

# Global ASR instance with concurrency control
asr_instance: Optional[StreamingASR] = None
asr_lock = asyncio.Lock()  # Protects model inference (GPU is single-threaded per model)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown."""
    global asr_instance
    
    # Startup: Load model
    model_path = os.environ.get(
        'ASR_MODEL_PATH',
        '/workspace/models/parakeet-ctc-0.6b-with-meta/parakeet-600m-encoder-tune-weight-0.5-batch-langfamily-balance-tokenizer-aggregated-new.nemo'
    )
    
    if os.path.exists(model_path):
        asr_instance = StreamingASR(model_path=model_path)
        print("ASR model loaded successfully!")
    else:
        print(f"WARNING: Model not found at {model_path}")
        print("Set ASR_MODEL_PATH environment variable to your model path")
    
    yield
    
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Streaming ASR API",
    description="Real-time speech recognition with metadata tags",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for WebSocket connections
sessions: Dict[str, np.ndarray] = {}


# Track active connections for monitoring
active_connections = {"websocket": 0, "http": 0}

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": asr_instance is not None
    }

@app.get("/status")
async def status():
    """
    Get server status including concurrency info.
    
    Note on concurrency:
    - Multiple WebSocket connections ARE supported
    - Each connection maintains its own audio buffer (isolated)
    - GPU inference is serialized via lock (one at a time)
    - Concurrent connections will queue for GPU access
    - Typical GPU latency: 50-200ms per inference
    
    For higher throughput:
    - Run multiple server instances behind a load balancer
    - Or use model batching (requires code changes)
    """
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
            "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }
    
    return {
        "status": "ok",
        "model_loaded": asr_instance is not None,
        "active_websocket_connections": active_connections["websocket"],
        "active_http_requests": active_connections["http"],
        "gpu": gpu_info,
        "concurrency_note": "GPU inference is serialized. Multiple connections queue for access.",
        "sample_rate": asr_instance.sample_rate if asr_instance else None
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.
    
    Accepts WAV files. Returns full transcription with metadata.
    """
    if asr_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read audio file
    content = await file.read()
    
    try:
        # Parse WAV
        with io.BytesIO(content) as f:
            with wave.open(f, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                audio_bytes = wav_file.readframes(n_frames)
                
        # Convert to numpy
        if sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]
        
        # Resample if needed
        if sample_rate != asr_instance.sample_rate:
            import scipy.signal
            audio = scipy.signal.resample(
                audio,
                int(len(audio) * asr_instance.sample_rate / sample_rate)
            )
        
        # Transcribe (with lock for thread safety)
        async with asr_lock:
            asr_instance.reset()
            result = asr_instance.process_chunk(audio, accumulate=False)
        
        return {
            "text": result.text,
            "text_only": result.text_only,
            "metadata": result.metadata,
            "duration": result.total_duration
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


@app.post("/stream/start")
async def stream_start(session_id: str = "default"):
    """Start a new streaming session."""
    if asr_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    sessions[session_id] = np.array([], dtype=np.float32)
    return {"status": "started", "session_id": session_id}


@app.post("/stream/chunk")
async def stream_chunk(
    session_id: str = "default",
    file: UploadFile = File(...)
):
    """
    Send an audio chunk for streaming transcription.
    
    Send raw PCM audio (16-bit, 16kHz, mono) or WAV.
    Returns incremental transcription.
    """
    if asr_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if session_id not in sessions:
        sessions[session_id] = np.array([], dtype=np.float32)
    
    content = await file.read()
    
    try:
        # Try to parse as WAV first
        try:
            with io.BytesIO(content) as f:
                with wave.open(f, 'rb') as wav_file:
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
        except:
            # Assume raw PCM int16
            audio = np.frombuffer(content, dtype=np.int16)
        
        # Accumulate
        audio_float = audio.astype(np.float32) / 32768.0
        sessions[session_id] = np.concatenate([sessions[session_id], audio_float])
        
        # Limit to 60 seconds
        max_samples = 60 * asr_instance.sample_rate
        if len(sessions[session_id]) > max_samples:
            sessions[session_id] = sessions[session_id][-max_samples:]
        
        # Transcribe accumulated audio (with lock for thread safety)
        async with asr_lock:
            asr_instance.audio_buffer = sessions[session_id]
            result = asr_instance.process_chunk(np.array([]), accumulate=True)
            # Restore buffer (process_chunk may have modified it)
            asr_instance.audio_buffer = sessions[session_id]
        
        return {
            "text": result.text,
            "text_only": result.text_only,
            "metadata": result.metadata,
            "duration": result.total_duration
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.post("/stream/end")
async def stream_end(session_id: str = "default"):
    """End streaming session and get final transcription."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "ended", "session_id": session_id}


@app.websocket("/listen/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming ASR.
    
    URL: /listen/{session_id}
    Production: wss://audio.whissle.ai/listen/{session_id}
    
    Send: binary audio chunks (raw PCM int16, 16kHz, mono)
    
    Receive: JSON with structured response:
        {
            "transcript": "clean text without any tags",
            "transcript_with_entities": "text with ENTITY_TYPE value END tags",
            "finalized": "committed text that won't change",
            "metadata": {
                "age": "AGE_18_30",
                "gender": "GENDER_MALE", 
                "emotion": "EMOTION_NEUTRAL",
                "intent": "INTENT_QUERY"
            },
            "entities": [
                {"type": "LOCATION", "value": "London", "raw": "ENTITY_LOCATION London END"},
                {"type": "DATE", "value": "tomorrow", "raw": "ENTITY_DATE tomorrow END"}
            ],
            "buffer_duration": 3.5,
            "is_stable": false
        }
    
    Configuration (send as JSON):
        {"config": {"buffer_duration": 12.0, "min_audio_for_transcription": 1.0}}
    
    Commands (send as text):
        - "RESET": Clear buffer and start fresh
        - "FINALIZE": Get final transcription with is_final=true
    """
    await websocket.accept()
    active_connections["websocket"] += 1
    
    if asr_instance is None:
        await websocket.send_json({"error": "Model not loaded"})
        active_connections["websocket"] -= 1
        await websocket.close()
        return
    
    # Configurable parameters (can be updated via config message)
    config = {
        "buffer_duration": 6.0,         # Left context - max audio to keep (seconds)
        "min_audio_for_transcription": 1.0,  # Don't transcribe until this much audio
        "commit_threshold": 0.6,        # Fraction of buffer to commit when trimming
    }
    
    # Per-session state
    session_buffer = np.array([], dtype=np.float32)
    finalized_text = ""              # Committed text from buffer overflow (won't change)
    last_buffer_len = 0              # Track buffer length at last transcription
    max_samples = int(config["buffer_duration"] * asr_instance.sample_rate)
    min_new_audio = int(1.5 * asr_instance.sample_rate)  # Min 1.5s new audio before re-transcribing
    
    # Cached transcription results
    last_clean_text = ""             # Pure transcript without any tags
    last_text_with_entities = ""     # Transcript with entity tags (no metadata)
    last_entities = []               # List of extracted entities
    last_metadata = {}               # Metadata dict (AGE, GENDER, EMOTION, INTENT)
    
    # VAD state
    vad_threshold = 0.02             # RMS energy threshold for speech detection (0.02 works well for most mics)
    silence_frames = 0               # Count consecutive silence frames
    max_silence_frames = 10          # After this many silence frames, stop updating
    has_speech = False               # Has any speech been detected this session?
    
    def is_speech(audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Simple energy-based VAD - returns True if audio contains speech."""
        if len(audio) == 0:
            return False
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio data
                audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                
                # VAD check - only process if there's speech
                chunk_has_speech = is_speech(audio_chunk, vad_threshold)
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                
                if chunk_has_speech:
                    silence_frames = 0
                    has_speech = True
                    print(f"[CHUNK] Speech detected - RMS: {rms:.4f}, chunk_len: {len(audio_chunk)}")
                else:
                    silence_frames += 1
                    if silence_frames <= 3:  # Only log first few silence frames
                        print(f"[CHUNK] Silence #{silence_frames} - RMS: {rms:.4f}")
                
                # Skip processing if prolonged silence (but keep connection alive)
                if silence_frames > max_silence_frames and has_speech:
                    # Send stable response without re-processing
                    await websocket.send_json({
                        "transcript": last_clean_text,
                        "transcript_with_entities": last_text_with_entities,
                        "finalized": finalized_text,
                        "metadata": last_metadata,
                        "entities": last_entities,
                        "buffer_duration": len(session_buffer) / asr_instance.sample_rate,
                        "is_stable": True,
                        "is_silence": True
                    })
                    continue
                
                # Add to buffer (only if speech or recent speech)
                if chunk_has_speech or silence_frames <= max_silence_frames:
                    session_buffer = np.concatenate([session_buffer, audio_chunk])
                
                # Check if we need to trim (sliding window)
                if len(session_buffer) > max_samples:
                    # BEFORE trimming: get transcription to find what to commit
                    async with asr_lock:  # Lock for thread-safe GPU access
                        pre_trim_raw = asr_instance.transcribe_audio(session_buffer)
                    pre_trim_text = asr_instance._get_text_only(pre_trim_raw)
                    
                    # Calculate how much text to commit
                    trim_amount = len(session_buffer) - max_samples
                    trim_fraction = trim_amount / len(session_buffer)
                    commit_target = int(len(pre_trim_text) * config["commit_threshold"])
                    
                    if commit_target > 0 and len(pre_trim_text) > commit_target:
                        import re
                        # Look for sentence boundary first
                        matches = list(re.finditer(r'[.!?]\s+', pre_trim_text[:commit_target + 30]))
                        if matches:
                            boundary = matches[-1].end()
                        else:
                            # Find word boundary
                            boundary = pre_trim_text.rfind(' ', 0, commit_target + 15)
                            if boundary > 0:
                                boundary += 1
                            else:
                                boundary = commit_target
                        
                        text_to_commit = pre_trim_text[:boundary].strip()
                        if text_to_commit and text_to_commit not in finalized_text:
                            finalized_text += (" " if finalized_text else "") + text_to_commit
                    
                    # Trim buffer - but DON'T force re-transcription of trimmed content!
                    # The committed text is already saved in finalized_text
                    session_buffer = session_buffer[-max_samples:]
                    
                    # Re-transcribe ONLY the trimmed buffer to get fresh cached values
                    # This is a one-time transcription after trim - WITH quality checks
                    async with asr_lock:
                        raw_text = asr_instance.transcribe_audio(session_buffer)
                    
                    # Quality check the trimmed transcription too
                    trim_valid = True
                    if '<unk>' in raw_text.lower() or 'unk>' in raw_text:
                        trim_valid = False
                        print(f"[TRIM-REJECT] <unk> in trimmed transcription")
                    if 'entity<' in raw_text.lower() or 'ity<' in raw_text.lower():
                        trim_valid = False
                        print(f"[TRIM-REJECT] decoder artifact in trimmed transcription")
                    
                    # Check for malformed entity tags
                    import re
                    if re.search(r'ENTITY_[A-Z]+[a-z]', raw_text):
                        trim_valid = False
                        print(f"[TRIM-REJECT] malformed entity in trimmed transcription")
                    
                    if trim_valid:
                        last_text_with_entities = asr_instance._get_text_only(raw_text)
                        last_clean_text = asr_instance._get_clean_text(raw_text)
                        last_entities = asr_instance._extract_entities(raw_text)
                        last_metadata = asr_instance._extract_metadata(raw_text)
                    # else: keep previous good values
                    
                    last_buffer_len = len(session_buffer)  # Mark as transcribed - no re-transcription until new audio
                
            elif "text" in message:
                try:
                    # Try to parse as JSON config
                    import json
                    msg_data = json.loads(message["text"])
                    if "config" in msg_data:
                        # Update configuration
                        for key in ["buffer_duration", "min_audio_for_transcription", "commit_threshold"]:
                            if key in msg_data["config"]:
                                config[key] = float(msg_data["config"][key])
                        max_samples = int(config["buffer_duration"] * asr_instance.sample_rate)
                        await websocket.send_json({"status": "config_updated", "config": config})
                        continue
                except:
                    pass
                
                cmd = message["text"].strip().upper()
                if cmd == "RESET":
                    session_buffer = np.array([], dtype=np.float32)
                    finalized_text = ""
                    last_buffer_len = 0
                    last_clean_text = ""
                    last_text_with_entities = ""
                    last_entities = []
                    last_metadata = {}
                    # Reset VAD state
                    silence_frames = 0
                    has_speech = False
                    await websocket.send_json({
                        "status": "reset",
                        "transcript": "",
                        "transcript_with_entities": "",
                        "finalized": "",
                        "metadata": {},
                        "entities": [],
                        "buffer_duration": 0,
                        "is_stable": False
                    })
                    continue
                elif cmd == "FINALIZE":
                    # User stopped speaking - return final results
                    await websocket.send_json({
                        "status": "finalized",
                        "transcript": last_clean_text,
                        "transcript_with_entities": last_text_with_entities,
                        "finalized": finalized_text,
                        "metadata": last_metadata,
                        "entities": last_entities,
                        "buffer_duration": len(session_buffer) / asr_instance.sample_rate,
                        "is_final": True,
                        "is_stable": True
                    })
                    continue
            
            # Only transcribe if:
            # 1. We have minimum audio
            # 2. Enough NEW audio has been added since last transcription
            # 3. There's been RECENT speech (not prolonged silence)
            min_samples = int(config["min_audio_for_transcription"] * asr_instance.sample_rate)
            new_audio_added = len(session_buffer) - last_buffer_len
            
            # Only transcribe during speech or shortly after (max 2 silence frames = ~0.36s grace period)
            has_recent_speech = chunk_has_speech or silence_frames <= 2
            
            should_transcribe = (
                len(session_buffer) >= min_samples and 
                new_audio_added >= min_new_audio and
                has_recent_speech  # Don't transcribe during prolonged silence
            )
            
            if should_transcribe:
                print(f"\n[TRANSCRIBE] Buffer: {len(session_buffer)/asr_instance.sample_rate:.2f}s, new_audio: {new_audio_added/asr_instance.sample_rate:.2f}s")
                
                # Lock ensures only one GPU inference at a time
                async with asr_lock:
                    raw_text = asr_instance.transcribe_audio(session_buffer)
                
                # Extract all components
                new_text_with_entities = asr_instance._get_text_only(raw_text)
                new_clean_text = asr_instance._get_clean_text(raw_text)
                new_entities = asr_instance._extract_entities(raw_text)
                new_metadata = asr_instance._extract_metadata(raw_text)
                
                print(f"[RAW] {raw_text[:150]}..." if len(raw_text) > 150 else f"[RAW] {raw_text}")
                print(f"[CLEAN] {new_clean_text[:100]}..." if len(new_clean_text) > 100 else f"[CLEAN] {new_clean_text}")
                
                # Quality check: don't replace good transcription with worse one
                is_better = True
                reject_reason = ""
                
                # Check for <unk> tokens in BOTH clean and entities text (always bad)
                if '<unk>' in raw_text.lower() or 'unk>' in raw_text:
                    is_better = False
                    reject_reason = "<unk> token in raw"
                
                # Check for decoder artifacts in raw text
                if 'entity<' in raw_text.lower() or 'ity<' in raw_text.lower():
                    is_better = False
                    reject_reason = "decoder artifact"
                
                # Check for broken entity tags (missing END or malformed)
                if 'ENTITY_' in raw_text and 'END' not in raw_text:
                    is_better = False
                    reject_reason = "incomplete entity tag (missing END)"
                
                # Check for malformed entity tags (ENTITY_TYPE runs into value without space)
                # Good: "ENTITY_LOCATION London END"  Bad: "ENTITY_CITYaris" or "ENTITY_LOCATIONLondon"
                import re
                malformed = re.search(r'ENTITY_[A-Z]+[a-z]', raw_text)
                if malformed:
                    is_better = False
                    reject_reason = f"malformed entity tag: {malformed.group()}"
                
                if last_clean_text and is_better:
                    # Check for significant length reduction (bad sign)
                    if len(new_clean_text) < len(last_clean_text) * 0.7:
                        is_better = False
                        reject_reason = f"length reduced ({len(new_clean_text)} < {len(last_clean_text)*0.7:.0f})"
                    
                    # Check for obvious word repetition (bad sign)
                    words = new_clean_text.lower().split()
                    if len(words) > 4:
                        repeats = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
                        if repeats >= 2:
                            is_better = False
                            reject_reason = f"word repetition ({repeats} repeats)"
                    
                    # Check for phrase duplication (same text appears twice)
                    if len(new_clean_text) > 30 and is_better:
                        half = len(new_clean_text) // 3
                        first_part = new_clean_text[:half].lower().strip()
                        if len(first_part) > 10 and first_part in new_clean_text[half:].lower():
                            is_better = False
                            reject_reason = "phrase duplication"
                
                # Only update if new transcription is better or first transcription
                if is_better:
                    print(f"[ACCEPT] Updated transcription")
                    last_text_with_entities = new_text_with_entities
                    last_clean_text = new_clean_text
                    last_entities = new_entities
                    last_metadata = new_metadata
                    last_buffer_len = len(session_buffer)  # Track we've processed this buffer
                else:
                    print(f"[REJECT] Keeping previous - reason: {reject_reason}")
                    last_buffer_len = len(session_buffer)  # Still mark as processed to avoid re-trying same buffer
                
                last_buffer_len = len(session_buffer)
            
            # Send structured response - easy for external apps to consume
            await websocket.send_json({
                # Clean transcript (no tags at all)
                "transcript": last_clean_text,
                
                # Transcript with entity tags inline (ENTITY_TYPE ... END)
                "transcript_with_entities": last_text_with_entities,
                
                # Finalized/committed text (won't change)
                "finalized": finalized_text,
                
                # Metadata as structured dict
                "metadata": last_metadata,
                
                # Entities as structured list
                "entities": last_entities,
                
                # Status
                "buffer_duration": len(session_buffer) / asr_instance.sample_rate,
                "is_stable": not should_transcribe and len(session_buffer) > 0
            })
            
    except WebSocketDisconnect:
        print(f"WebSocket {session_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        active_connections["websocket"] -= 1
        await websocket.close()


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model", help="Path to .nemo model file")
    
    args = parser.parse_args()
    
    if args.model:
        os.environ['ASR_MODEL_PATH'] = args.model
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
