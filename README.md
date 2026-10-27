# Local Voice AI Project

This project explores a local voice pipeline step by step. The current implementation transcribes a recorded audio file using faster-whisper. An LLM and text-to-speech are planned for later stages.

## Current stage: speech-to-text

`src/models.py` loads Whisper `large-v3-turbo`, transcribes an MP3 file, and prints the detected language, its confidence score, and timestamped text segments. It is configured to run on an NVIDIA GPU with CUDA and FP16.

The script currently calls transcription twice on the same file. It also creates a `BatchedInferencePipeline`, but does not use it for transcription.

## Setup and run

From the project root in Windows PowerShell, activate your existing virtual environment and install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current script requires a compatible NVIDIA GPU and CUDA libraries configured for faster-whisper. The model may download on its first run. See the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper) for GPU requirements.

Place a recording at `sample/audio_2.mp3`, or change the path in `src/models.py` to your own audio file. The `sample/` folder is ignored by Git, so recordings are not included in the repository.

Run from the project root:

```powershell
python -m src.models
```

If you encounter the `metadata_errors` error with PyAV 19 and faster-whisper 1.2.1, the [reported workaround](https://github.com/SYSTRAN/faster-whisper/issues/1589) is `python -m pip install "av<19"`.

## Planned stages

- **LLM:** Use a local Llama 3.2 model through Ollama to generate a response from the transcript.
- **Text-to-speech:** Use Kokoro-82M to turn the response into audio.
- **Integration:** Connect audio input, transcription, response generation, and playback.

These stages are not yet implemented as a working pipeline. Docker configuration also needs updating before it can run the current STT script.

## Learning reference

The project uses [Voice Agents from Scratch](https://github.com/pguso/voice-agents-from-scratch) as a learning resource.
