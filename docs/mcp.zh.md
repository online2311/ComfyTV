[English](mcp.md) | **简体中文**

# Agent 接入(MCP)

> ComfyTV 内置 MCP 服务,让 AI agent 读取并驱动你的画布:搭建节点图、跑渲染、阻塞等待、用真视觉检查结果、编辑原生 ComfyUI 图、管理工作流配置 — 共 46 个工具,外加把已装的 [Agent Skills](skills.zh.md) 作为 MCP prompts 提供。

## 是什么

ComfyTV 在你的 ComfyUI 服务器上开放 [Model Context Protocol](https://modelcontextprotocol.io) 端点 `POST /comfytv/mcp`。任何支持 MCP 的 agent(Claude Code,或任何会说 streamable HTTP JSON-RPC 的客户端)都能连上来,把 ComfyTV 当成一套工具使用。

有页面在镜像时,写工具仍由那一页经 websocket 执行,和以前一样。**没有打开的页面时**,stage / 图的写入落到服务器上持久化的无头图,`run_stage` / `graph_run` 直接进 ComfyUI 自己的 prompt 队列,立刻返回 job id,不再等浏览器超时。场景、导演台、图层编辑器和视口命令仍然需要打开的页面。

## 启用

MCP **默认关闭**。在 ComfyTV 侧边栏打开 **设置 → Agent 与 MCP**,开启**启用 MCP 服务**。(内嵌 [Bot](bot.zh.md) 还需要它自己的开关。)

## 连接客户端

Claude Code:

```bash
claude mcp add --transport http comfytv http://127.0.0.1:8188/comfytv/mcp
```

Codex:把服务器加到 `~/.codex/config.toml`:

```toml
[mcp_servers.comfytv]
url = "http://127.0.0.1:8188/comfytv/mcp"
```

其他 MCP 客户端:用 streamable HTTP 指向同一 URL。处理器本身仍是无状态的,但服务器会在 `initialize` 时签发一个 `Mcp-Session-Id`,以兼容期望 streamable-HTTP 会话管理的客户端。不在 ComfyUI 之上附加鉴权 — 8188 端口的网络暴露请自行斟酌。

## 工具目录

**读取与发现**

| 工具 | 作用 |
|---|---|
| `server_info` | 版本、stage 数、镜像状态、近期错误 |
| `projects` | 项目列表/详情 |
| `stage_catalog` | 全部 stage 类型与工作流 kind |
| `list_workflows` | 各 kind 的工作流库 |
| `get_canvas` | 画布实时快照(节点、prompt、运行状态) |
| `get_stage` | 单节点全量详情:所有 widget 值、双向连线、参考、警告 |
| `outputs` | 各 stage 的渲染历史 |
| `assets` / `resources` / `entries` | 资产库、LUT/字体/音色文件、提示词条目 |
| `jobs` / `exec_errors` / `servers` | 远程任务、近期错误、机器列表 |
| `node_info` | 搜索 ComfyUI 已装节点类并查看其输入/输出 |
| `skill` | 列出已装的 [Agent Skills](skills.zh.md) 并读取其指令 — 工具自身的 description 内嵌一份实时技能索引 |

**搭建与运行**

| 工具 | 作用 |
|---|---|
| `add_stage` / `set_stage` | 建节点、配节点:prompt、工作流、widget、资产参考、目标机器 |
| `connect_stages` | 连线(类型自动匹配) |
| `run_stage` | 排队运行。有页面时由页面执行;无页面时立刻返回 `job_id` |
| `wait_stage` | 阻塞等到出结果或报错。无页面时改等无头任务 |
| `queue_status` | 查一个无头任务的状态、进度和 `/view` 产出,不阻塞 |
| `list_queue` | 列出无头队列(不是远程机器的 `jobs`) |
| `resume_queue` | 先查 `/queue` 和 `/history/{prompt_id}`，只有 ComfyUI 完全没有这条 prompt 的记录时才重新提交。启动时默认不自动续跑（`headless-auto-resume` 默认关） |
| `cancel_stage` | 有页面时停止该次运行。无页面时只丢掉还没提交的任务;已经在 ComfyUI 队列里的不中断 |
| `remove_stage` | 删节点 |
| `arrange_canvas` | 整理画布布局(行/列/网格) |

**原生 ComfyUI 图**

| 工具 | 作用 |
|---|---|
| `graph_get` | 读原生画布图 — 所有节点、widget、连线、分组,不限 ComfyTV 节点 |
| `graph_edit` | 批量编辑原生图:加/删/克隆节点、设 widget、连线、分组、折叠、bypass、打包/解包子图 — 整批一个撤销步 |
| `graph_run` | 排队原生图。有页面时最多等 `wait_s`。无页面时传 `prompt`(API 格式)或 `workflow`(UI 格式),立刻返回 `job_id` |
| `workflow_create` | 用 API 格式的图为某个 stage kind 新建一个工作流文件 |
| `canvas_command` | 白名单原生命令:撤销/重做、保存、适配视图、中断、刷新节点定义 |
| `canvas_focus` | 选中节点并把用户视口平滑移过去 — 展示你改了什么 |

**看见与评判**

| 工具 | 作用 |
|---|---|
| `view_image` | 返回图像本体(缩放后),agent 能亲眼看 |
| `fx_preview` | 把视频 ~1.2 秒窗口过一遍单个 FX 节点的真滤镜链并直接看中间帧 — 全片渲染前廉价迭代调参 |
| `media_probe` | 视频时长/帧率/分辨率/有无音轨 |
| `media_frame` | 从视频抽一帧,返回 PNG URL |
| `media_waveform` | 渲染音频波形图 |
| `media_timeline` | 视频区间的胶片条+波形合成图,静音段自动检出当剪点候选 |
| `pick_output` | 从多候选出图里选定一张 |

**配置**

| 工具 | 作用 |
|---|---|
| `workflow_get` / `workflow_edit` | 读工作流的输入绑定与节点清单;绑定/解绑 stage 值到工作流输入,按真实图校验 |
| `stage_params` | 定义自定义参数,再以 `option:<key>` 绑定 |
| `asset_edit` | 把产出存入资产库、改名、归类 |
| `series_library` | 读写系列目录里的 `library.yaml`（稳定代号 `ID@vN#view`、锁定、新版本、解析到文件、JSON manifest 导入）。状态是 `draft`、`candidate`（已挑未锁）或 `locked`。默认目录是 `<input>/comfytv/series/<series>`（`COMFYTV_ASSET_ROOT` 可改 `comfytv` 这一层）。系列名不能是空串、绝对路径，也不能含 `/`、`\` 或 `..`。`library.yaml` 里的 `spark_root` 会跟着走，重定向之后的目录仍必须在资产根内。`legacy_ids` 可以是字符串列表，也可以是按视图的映射 |
| `character_studio` | 同一份库上的角色档案：别名、`legacy_ids`（列表或按视图的映射）和 `P01_` 名、脸模导入、H3 Image 合成定妆/角色卡/派生/精修、s2pro 试音。成片写到 `<output>/comfytv/projects/<project>/`（`COMFYTV_OUTPUT_ROOT` 可改输出侧的 `comfytv` 目录）。静图和试音返回 `job_id`。角色卡默认不进视频参考槽 |

`resolve`（单个代号、代号列表，或 shot 模式）返回代号上写明的视图。省略 `#view` 时，角色、造型、服装在有文件的情况下用 `#front`。它不会额外挂上 `#headshot`。series-kit 的 `compile` 会在 `#front` 之外再挂角色的 `#headshot`；要拿这张图，对 `resolve` 传入 `CHR_…@vN#headshot`。shot 模式只解析镜头上已经写下的代号，STY、LOC、PRP 也一样。

**富编辑器**

| 工具 | 作用 |
|---|---|
| `scene_get/edit/capture/record` | 驱动 Scene 3D:角色、灯光、相机、动画 clip、相机路径与多镜导演时间轴(add_shot/patch_shot/set_path) |
| `director_get/director_edit` | 读写导演台 clip 时间线 |
| `storyboard` | 读取、替换或校验分镜阶段的完整镜头文档（`shots.json` 字段和自定义键）。`get_stage` 对很长的 `storyboard_data` 只给预览。`validate` 的整集总时长按每镜 `duration_s` 相加，`cut_to` 只和该镜时长比较。镜头少到即使每镜 15 秒也达不到 `target_duration_s` 时（单镜或很短的片段），不再拿整集目标来比。可选的分镜提示词字段（`aspect_ratio`、`refs.slot`、`subjects`、H3 的 `summary_tasks` / retention / `align_s`）出现时才检查；`strict` 额外要求开场画面、结尾画面、`end_hold` 和至少一段 segment。图片 `src` 是 ComfyTV 资产代号或 `episode:` 路径。`workflow` 决定图片上限（FL2VA 为 2，Ref2VA 为 9） |
| `episode_batch` | 按分镜为每镜准备或提交一个 VideoStage。`plan` 用系列库解析资产代号和 `episode:` 首尾帧，给出可复现的逐镜 seed，选择 Ref2VA 或 FL2VA，并按配置挂上台词音频；不调用模型。本该用 FL2VA、又有台词、且该 FL2VA 工作流不接上游音频时，plan 和 run 都改成 Ref2VA（`reason` 为 `dialogue_downgrade`），除非镜上写了 `video_mode=FL2VA`。`run` 在 GET /queue 空闲后一次只提交一镜，已完成的镜会跳过，不会中断队列 |
| `script_writer` | 剧本会话在 `<ComfyUI input>/comfytv/projects/<project>/script/`（`COMFYTV_ASSET_ROOT` 替换这一层 `comfytv`）。传入的 `root` 必须落在该 `projects/` 目录里。生成出的成片目录是 `<ComfyUI output>/comfytv/projects/<project>/`（`COMFYTV_OUTPUT_ROOT`）。版本、选段改写，以及带 sha256 的清单导入。不写 `library.yaml`。不含 `to_shots` |
| `segment_reshoot` | 计划、执行、列出、采用、回滚或登记保护一条镜头内部的纯音频重拍。`plan` 不提交提示。`adopt` 和 `rollback` 的 `approved_by` 必须是张景。记录在 `<input>/comfytv/projects/<project_id>/reshoots/<shot_id>/takes.json` 和 `take_<n>_r<k>.json`（`COMFYTV_ASSET_ROOT` 可改输入侧的 `comfytv`）。新成片在 `<output>/comfytv/projects/<project_id>/reshoot/<shot_id>/take_<n>_r<k>.mp4`（`COMFYTV_OUTPUT_ROOT` 可改输出侧的 `comfytv`）。`protect` 把相对该项目输出目录的通配符（如 `rough_cut*.mp4`）写进 `protected.json`。匹配到的文件不会被覆盖或删除 |

## 关键套路

**生产循环。**`stage_catalog` → `add_stage`(prompt、工作流、`asset_refs`)→ `connect_stages` → `run_stage` → `wait_stage` → `outputs` → `view_image` 质检 → 迭代。

**不轮询的等待。**`wait_stage` 在服务器侧阻塞(单次默认 25 秒,上限 170 秒),结果一落地立即返回。超时返回 `after_output_id` — 带着它再调就继续等。总时长无上限,长渲染多续调几次而已。

**@ 序数是从 1 起的位置。**每个节点按类型维护一张有序媒体表 — 连线输入和 `asset_refs` 在同一张表里,顺序以 `get_stage` 返回的 `media` 为准。`@image_1` / `@video_1` 这类 token 按位置寻址(`@image_1` 就是第一张图)。删除或重排媒体时,节点 prompt 里的 token 会自动重新编号;`set_stage` 的 `media_order` 用来重排,`connect_stages` 会返回这根线落在第几位。越界 token 展开为空 — 工具结果会带警告提醒。

**真视觉。**`view_image` 是唯一返回像素的工具。视频质检:先 `media_frame` 抽帧,再 `view_image` 亲眼看。别靠文件名猜图。

**给新 link 的工作流配绑定。**`workflow_get` 看 API 图节点清单;`workflow_edit` 用 `option:seed`、`computed:width` / `computed:height`(尺寸引擎)、`main_prompt`、`upstream_image:value[0]`、`literal:...` 这些来源绑定。工作流需要 ComfyTV 没有默认键的参数时,先用 `stage_params` 建一个,再绑 `option:<它的 key>`。`reset_to_preset` 是后悔药。

**多机。**`servers` 列出配置的机器和实时负载;`set_stage {server: <id>}` 把某节点的运行路由过去,结果落回本机。

**两层图。**stage 工具(`add_stage`/`set_stage`/…)工作在 ComfyTV 产品层;`graph_*` 家族直接编辑底下的原生 ComfyUI 图 — 任何插件的任何节点。有现成 stage 就用 stage;要搭裸管线就下到 `graph_edit` + `graph_run`,再用 `canvas_focus` 给用户看你改了哪。

**技能。**用户装了 [Agent Skills](skills.zh.md) 时,`skill` 工具的 description 会列出它们;任务命中就先读再照做。每个启用的技能同时暴露为一条 **MCP prompt** — 在 Claude Code 里表现为 `/mcp__comfytv__<技能名>` 斜杠命令,可显式调用。

## 无头队列

没有打开的 ComfyTV 页面时:

1. `add_stage` / `set_stage` / `connect_stages` / `graph_edit` 改的是按项目存在数据库里的图(`comfytv_headless_graphs`)。`graph_get` 读的就是它。页面正在镜像时,那一页仍是该项目的真相。
2. `run_stage` 和 `graph_run` 写入 `comfytv_headless_jobs` 并立刻返回 `{job_id, status: "queued"}`。`graph_run` 可直接传要 POST 给 `/prompt` 的 API 图。
3. 工作线程**一次只提交一个**任务到 ComfyUI 进程内的 prompt 队列(和 `GET /queue` 是同一份),而且仅当该队列里既没有 running 也没有 pending。这条路径不调用 `/interrupt` 或 `/free`。
4. 用 `queue_status` / `list_queue` 看 `queued` / `running` / `done` / `error`、进度和产出。`created_at` / `updated_at` / `started_at` / `finished_at` 是带偏移的 UTC ISO-8601（`+00:00`）。`started_at` 是交给 ComfyUI 的时刻；`finished_at` 在 done、error 或 cancel 时写入。`completed_nodes` 来自执行进度，成功时包含 prompt 里的每个节点。不要靠固定超时干等。
5. 任务状态落在 sqlite 里。启动时先查 `GET /queue`（running 和 pending）和 `GET /history/{prompt_id}`。还在队列里的重新挂上并继续等；已经在 history 里的收回产出，不再提交。ComfyUI 完全没有记录时**不会**自动再提交，除非调用 `resume_queue`，或打开设置 `headless-auto-resume`（默认关）。再次提交时会丢掉已经存下产出的 stage，下游输入改接到那份产出。对 `error` 任务传入 `job_id` 会在同样的检查之后再排队。

新表由现有的启动 `create_all` 建出。这条路径不另开监听端口。

在 ComfyUI v0.45.0(commit `989d93d`)上、不打开浏览器验证:MCP 开着、不要打开 ComfyTV 页面; `server_info` 里 `write_tools_need_open_tab` 为 false; `graph_run` 传一个小 API `prompt`,应立刻得到 `job_id`; `GET /queue` 里来自 ComfyTV 的任务同时不超过一个;第二个 `graph_run` 在 `list_queue` 里保持 `queued`,直到第一个离开 `/queue`;长任务用 `queue_status` 等到 `done`。中途重启后任务变为 `interrupted`（自动续跑默认关）。`resume_queue` 只有在 `/queue` 和 `/history/{prompt_id}` 都没有这条 prompt 时才补交未完成的 stage；若 prompt 仍在跑，则只重新挂上，不再提交第二条。

## 前置条件与行为说明

- 页面开着且镜像还新鲜时,写和跑仍走那一页。镜像在首次 MCP 调用后懒激活(刚连上时 `get_canvas` 等 ~10 秒重试)。
- 没有页面时走上面的无头队列。`queue_status` / `list_queue` 是无头任务;`jobs` 仍是远程机器列表。
- MCP 客户端会按会话缓存 `tools/list` — 服务器加了新工具后需重连。
- 客户端若有单次工具调用超时,用 `queue_status` 轮询,不要把等待放进 `graph_run`。`wait_stage` 仍按 `timeout_s`(默认 90,最大 170)切片,可以接着调。

## 另见

- [ComfyTV Bot](bot.zh.md) — 基于同一套工具的内嵌聊天代理
- [Agent Skills](skills.zh.md) — 经 `skill` 工具和 MCP prompts 提供的指令包
- [自定义工作流](custom-workflows.zh.md) — 绑定是什么、手工怎么配
- [侧边栏](sidebar.zh.md) — 带 MCP 开关的设置面板
