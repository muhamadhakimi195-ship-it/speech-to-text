from faster_whisper import WhisperModel,  BatchedInferencePipeline
import time
import logging

model_size = "large-v3-turbo" 
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
start_time = time.perf_counter()
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16") // GPU with INT8

# model = WhisperModel(model_size, device="cpu", compute_type="int8") //CPU with INT8
batched_model = BatchedInferencePipeline(model=model)
segments, info = model.transcribe("./sample/audio_2.mp3", 
                                  beam_size=16, 
                                  temperature=0.0,
                                  condition_on_previous_text=False,
                                  vad_filter=True,
                                  vad_parameters=dict(min_silence_duration_ms=500))

#end time
end_time = time.perf_counter()

# duration
duration = end_time - start_time


print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# print transcribed segments
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


print(f"\n\nExecution time: {duration:.6f} seconds")