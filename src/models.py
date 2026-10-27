from faster_whisper import WhisperModel,  BatchedInferencePipeline

model_size = "large-v3-turbo" 

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16") // GPU with INT8

# model = WhisperModel(model_size, device="cpu", compute_type="int8") //CPU with INT8
batched_model = BatchedInferencePipeline(model=model)
segments, info = model.transcribe("./sample/audio_2.mp3", beam_size=16)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

segments, _ = model.transcribe("./sample/audio_2.mp3")
segments = list(segments)  # The transcription will actually run here.