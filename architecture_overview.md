# Architecture overview

## Implemented: recorded audio to text

```text
MP3 recording -> faster-whisper / Whisper large-v3-turbo -> terminal output
                                                            |- detected language and confidence
                                                            +- timestamped text segments
```

`src/models.py` loads the model with CUDA and FP16, then reads `sample/audio_2.mp3` relative to the project root. The recording must be supplied locally because `sample/` is ignored by Git.

The script currently makes two transcription calls for the same audio. A batched inference object is initialized, but the transcription calls use the underlying model directly.

## Planned: voice pipeline

```text
Audio input -> faster-whisper -> Ollama / Llama 3.2 -> Kokoro -> audio playback
```

The LLM, TTS, microphone input, and playback are future work; they are not connected to the current STT script.

## Project files

| Path | Purpose |
| --- | --- |
| `src/models.py` | Recorded-audio transcription experiment |
| `debug/test_dep.py` | Dependency and audio-device checks |
| `requirements.txt` | Dependencies for the planned project stack |
