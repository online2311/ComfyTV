[English](README.md) | **简体中文**

# `storyboard/` 工作流

这个目录下的工作流出现在 **Storyboard** 下拉框。接收一段故事/前提提示词,产出一份结构化镜头表(**不是图** , 逐 shot 的图在下游的 Shot Images stage 出)。

## stage 提供的输入

- **提示词** , 用户的故事前提 / brief。

## 工作流需要包含

- 一个文本节点,第 0 个输出是模型原文。StoryboardStage 会自己把前提包成 16 字段 ChatML,并解析回复。能解析的回复是 `镜号:` 字段表(半角或全角冒号)、JSON 对象 `{"shots": [...]}`，或短剧管线的 `shots.json`（这份文档原样保留，包括 16 字段以外的键）。
- 提示词输入接 LLM 的 prompt 端。关掉模型自带的对话模板(`use_default_template=false`),避免把 stage 的 ChatML 再包一层。
- 结果走 `graph_output_first` 取该节点的第 0 个输出。

字段名是 `nodes/stages/common/storyboard.py` 里 `_shape_storyboard_from_llm` 产出的 16 个键。

加自己的工作流见 [docs/custom-workflows.zh.md](../../docs/custom-workflows.zh.md);具体绑定在画布上选中 stage 后通过左侧 **ComfyTV** 侧边栏配,详见 [docs/sidebar-config-editor.zh.md](../../docs/sidebar-config-editor.zh.md)。

## 当前内置

`local-qwen3-storyboard` — 本地 Qwen3 4B,走核心节点 `TextGenerate`。预设把 stage 提示词原样送进去,并从节点 7 读生成的字符串。需要 `qwen_3_4b.safetensors`。
