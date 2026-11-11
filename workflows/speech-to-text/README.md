**English** | [简体中文](README.zh.md)

# `speech-to-text/` workflows

Workflows in this folder appear in the **Subtitles · Speech-to-Text** dropdown (`ComfyTV.SubtitleGenStage`). The stage passes one upstream video or audio clip and expects SRT text back.

## Shipped workflow: Local Whisper large-v3

`local-whisper-large-v3.json` shows up as **Local Whisper large-v3**. The graph is a single `ComfyTV.WhisperTranscribe` node. Slot 0 is SRT, which the preset reads with `graph_output_first`.

`SubtitleGenStage` puts the upstream `/view?` URL in `option:source`. The preset binds that to the node's `media` input. The node decodes audio with soundfile when it can, and with PyAV for video (and as a fallback). It does not use torchaudio.

### Model path

The node never downloads. It looks for a Hugging Face snapshot of `openai/whisper-large-v3` (a directory that contains `config.json`) in this order:

1. The `model_dir` widget, when it is not empty.
2. The environment variable `COMFYTV_WHISPER_MODEL`.
3. A snapshot under ComfyUI `models/whisper`, then `models/stt`. A directory whose name contains `large-v3` wins.
4. `/home/nodecloud/whisper-v3/models/whisper-large-v3`, when that directory exists.

Loading is `AutoModelForSpeechSeq2Seq.from_pretrained(..., dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True).to("cuda")` plus `AutoProcessor.from_pretrained(..., local_files_only=True)`. Audio longer than 30 seconds uses the model's own `generate()` long-form loop (`return_timestamps=True`), not an ASR pipeline `chunk_length_s`.

The model stays cached in the ComfyUI process. Turn on **unload** to drop it and call `torch.cuda.empty_cache()`. The node does not call ComfyUI `/interrupt` or `/free`.

Defaults are aimed at a Chinese short drama: language `zh`, task `transcribe`, timestamps `segment`.

### Manual check on Spark

Use a few seconds of Chinese dialogue, for example a line such as 「你好，这是一段短剧对白。」 saved as `sample-zh.wav` or a short mp4.

1. Confirm the snapshot directory exists, or export `COMFYTV_WHISPER_MODEL` to it.
2. Restart ComfyUI so the node and the workflow are registered.
3. **Node:** add **Whisper Transcribe**, set `media` to the absolute path, leave language on `zh`, run. The SRT output should contain the line with a cue time.
4. **Stage:** add **Subtitles · Speech-to-Text**, pick **Local Whisper large-v3**, wire the clip into `video` or `audio`, press Run. The text output is SRT. Wire it into **Subtitles** (`subs_text`) to burn it onto the picture.

No new pip packages are required. Do not install `accelerate`, `librosa`, `faster-whisper`, `ctranslate2`, or `openai-whisper` for this node.
