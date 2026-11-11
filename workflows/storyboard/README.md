**English** | [简体中文](README.zh.md)

# `storyboard/` workflows

Workflows in this folder appear in the **Storyboard** dropdown. Take a story / premise prompt; produce a structured shot list (**not images** — per-shot image generation happens downstream in the Shot Images stage).

## Stage inputs

- **Prompt** — the user's premise / brief.

## What your workflow needs

- A text node whose slot 0 is the raw model string. StoryboardStage already wraps the premise in the 16-field ChatML prompt and parses the reply. Accepted replies are the `镜号:` field list (ASCII or fullwidth colon) or a JSON object `{"shots": [...]}`.
- The prompt input wired to the LLM's prompt port. Turn the model's own chat template off (`use_default_template=false`) so the stage's ChatML is not wrapped a second time.
- Result type `graph_output_first` pointing at that node's slot 0.

Field names are the 16 keys produced by `_shape_storyboard_from_llm` in `nodes/stages/common/storyboard.py`.

To add your own workflow see [docs/custom-workflows.md](../../docs/custom-workflows.md); to configure per-node bindings, select the stage on the canvas and open the left **ComfyTV** sidebar — see [docs/sidebar-config-editor.md](../../docs/sidebar-config-editor.md).

## What's here today

`local-qwen3-storyboard` — local Qwen3 4B via core `TextGenerate`. The preset sends the stage prompt as-is and reads the generated string from node 7. Requires `qwen_3_4b.safetensors`.
