[English](README.md) | **简体中文**

# `speech-to-text/` 工作流

这个目录下的工作流出现在 **语音转字幕**（`ComfyTV.SubtitleGenStage`）的下拉框里。stage 传入一条上游视频或音频，期望拿回 SRT 文本。

## 自带工作流：Local Whisper large-v3

`local-whisper-large-v3.json` 在下拉框里的名字是 **Local Whisper large-v3**。图里只有一个 `ComfyTV.WhisperTranscribe` 节点。第 0 个输出是 SRT，preset 用 `graph_output_first` 读取。

`SubtitleGenStage` 把上游的 `/view?` 地址放进 `option:source`，preset 把它绑到节点的 `media`。音频文件优先用 soundfile 读，视频（以及 soundfile 打不开的文件）用 PyAV 解出音轨。不用 torchaudio。

### 模型路径

节点不会下载。它按下面的顺序找 `openai/whisper-large-v3` 的 Hugging Face 快照（目录里要有 `config.json`）：

1. `model_dir` 不为空时用它。
2. 环境变量 `COMFYTV_WHISPER_MODEL`。
3. ComfyUI 的 `models/whisper`，然后 `models/stt`。目录名里带 `large-v3` 的优先。
4. 该目录存在时，用 `/home/nodecloud/whisper-v3/models/whisper-large-v3`。

加载方式是 `AutoModelForSpeechSeq2Seq.from_pretrained(..., dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True).to("cuda")`，以及 `AutoProcessor.from_pretrained(..., local_files_only=True)`。超过 30 秒的音频走模型自己的 `generate()` 长音频循环（`return_timestamps=True`），不用 pipeline 的 `chunk_length_s`。

模型会缓存在 ComfyUI 进程里。打开 **unload** 会丢掉缓存并调用 `torch.cuda.empty_cache()`。节点不会请求 ComfyUI 的 `/interrupt` 或 `/free`。

默认值按中文短剧来：语言 `zh`，任务 `transcribe`，时间戳 `segment`。

### 在 Spark 上手工试一段中文

准备几秒中文对白，例如「你好，这是一段短剧对白。」存成 `sample-zh.wav`，或者一条短 mp4。

1. 确认快照目录在磁盘上，或把 `COMFYTV_WHISPER_MODEL` 指到它。
2. 重启 ComfyUI，让节点和工作流注册进去。
3. **节点：** 添加 **Whisper Transcribe**，`media` 填绝对路径，语言保持 `zh`，运行。SRT 输出里应有这句对白和时间码。
4. **Stage：** 添加 **语音转字幕**，工作流选 **Local Whisper large-v3**，把片子接到 `video` 或 `audio`，按 Run。文本输出是 SRT。再接到 **字幕** 节点的 `subs_text` 可以烧到画面上。

不需要新的 pip 包。不要为这个节点安装 `accelerate`、`librosa`、`faster-whisper`、`ctranslate2` 或 `openai-whisper`。
