import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5
audio_queue = queue.Queue()

model = WhisperModel(
    "large-v3-turbo", device="cuda", compute_type="float16"
)


def callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


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
            chunk = audio_queue.get()
            chunks.append(chunk)
            sample_count += len(chunk)

            if sample_count < SAMPLE_RATE * CHUNK_SECONDS:
                continue

            audio = np.concatenate(chunks)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            # print(
            #     f"Captured {len(audio) / SAMPLE_RATE:.1f}s | "
            #     f"Peak: {np.max(np.abs(audio)):.5f}",
            #     flush=True,
            # )
            chunks = []
            sample_count = 0

            segments, info = model.transcribe(
                audio,
                task="transcribe",
                language="en",
                beam_size=5, 
                temperature=0.1, 
                vad_filter=True
            )
            for segment in segments:
                if rms > 0.005:
                    print(segment.text.strip(), flush=True)


except KeyboardInterrupt:
    print("\nStopped.")