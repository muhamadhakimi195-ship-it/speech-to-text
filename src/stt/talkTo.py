import queue
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from src.llm.rag import rag_chain

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
audio_queue = queue.Queue()

# model = WhisperModel(
#     "large-v3-turbo", device="cuda", compute_type="float16"
# )
model = WhisperModel("tiny", device="cuda", compute_type="float16") #CPU with INT8
def callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())

def listen_until_pause():
    chunks = []
    speaking = False
    silent_samples = 0

    silence_threshold = 0.005
    pause_seconds = 0.5

    while True:
        chunk = audio_queue.get()
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if rms > silence_threshold:
            speaking = True
            silent_samples = 0
        elif speaking:
            silent_samples += len(chunk)

        if speaking:
            chunks.append(chunk)

        if speaking and silent_samples >= SAMPLE_RATE * pause_seconds:
            return np.concatenate(chunks)

try:
    with sd.InputStream(
        device=1,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        print("Listening... Press Ctrl+C to stop.", flush=True)
        chunks = []
        sample_count = 0

        while True:
            audio = listen_until_pause()

            segments, info = model.transcribe(
                audio,
                task="transcribe",
                language="en",
                beam_size=3,
                temperature=0.0,
                vad_filter=True,
            )

            question = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

            if not question:
                continue

            print(f"You: {question}", flush=True)

            if question.lower().rstrip(".!?") in ("end", "exit"):
                break
            start_time = time.perf_counter() 
            print("Assistant: ", end="", flush=True)

            for text in rag_chain.stream(question):
                print(text, end="", flush=True)

            print()
            #end time
            end_time = time.perf_counter()
    
            # duration
            duration = end_time - start_time
            print(f"{duration:.6f} seconds")

except KeyboardInterrupt:
    print("\nStopped.")