**English** | [简体中文](mcp.zh.md)

# Agent access (MCP)

> ComfyTV ships an MCP server so AI agents can read and drive your canvas: build node graphs, run stages, wait for renders, inspect results with real vision, edit the native ComfyUI graph, and manage your workflow configuration — including a headless queue for when no page is open — plus installed [Agent Skills](skills.md) served as MCP prompts.

## What it is

ComfyTV exposes a [Model Context Protocol](https://modelcontextprotocol.io) endpoint at `POST /comfytv/mcp` on your ComfyUI server. Any MCP-capable agent (Claude Code, or anything speaking streamable HTTP JSON-RPC) can connect and use ComfyTV as a set of tools.

When a ComfyTV page is open and mirroring, that page is still the live canvas: write tools are relayed to it over the websocket, exactly as before. **With no page open, stage and graph writes are applied on the server** to a persisted headless graph, and `run_stage` / `graph_run` queue work on ComfyUI's own prompt queue. They return a job id immediately instead of waiting out a browser timeout. `series_library` only reads and writes `library.yaml` on disk. `storyboard` `validate` is server-side; `get` and `set` use the headless graph's `storyboard_data` widget when no page is open. Scene, director, and layer editors, and viewport commands, still need a page.

## Enabling

MCP is **off by default**. In the ComfyTV sidebar open **Settings → Agent & MCP** and turn on **Enable MCP server**. (The embedded [Bot](bot.md) additionally requires its own switch.)

## Connecting a client

For Claude Code:

```bash
claude mcp add --transport http comfytv http://127.0.0.1:8188/comfytv/mcp
```

For Codex, add the server to `~/.codex/config.toml`:

```toml
[mcp_servers.comfytv]
url = "http://127.0.0.1:8188/comfytv/mcp"
```

Any other MCP client: point it at the same URL with the streamable HTTP transport.
The handlers are stateless, but the server issues an `Mcp-Session-Id` on
`initialize` for clients that expect streamable-HTTP session management. No auth
is added on top of your ComfyUI instance, so treat network exposure of port 8188
accordingly.

## The tool catalog

**Read & discover**

| Tool | What it does |
|---|---|
| `server_info` | Version, stage count, mirror state, recent errors |
| `projects` | List/get projects |
| `stage_catalog` | Every stage type and workflow kind |
| `list_workflows` | Workflow library per kind |
| `get_canvas` | Live canvas snapshot (nodes, prompts, run states) |
| `get_stage` | One stage in full detail: every widget value, connections both ways, references, warnings |
| `outputs` | Render history per stage |
| `assets` / `resources` / `entries` | Asset library, LUT/font/soundfont files, prompt snippets |
| `jobs` / `exec_errors` / `servers` | Remote jobs, recent errors, machine list |
| `node_info` | Search ComfyUI's installed node classes and inspect their inputs/outputs |
| `skill` | List installed [Agent Skills](skills.md) and read their instructions — the tool's own description carries a live index of what's installed |

**Build & run**

| Tool | What it does |
|---|---|
| `add_stage` / `set_stage` | Create and configure stages: prompt, workflow, widgets, asset references, target server |
| `connect_stages` | Wire outputs to inputs (auto-matches types) |
| `run_stage` | Queue a run. With a page open, the page runs it. With no page, returns a `job_id` immediately |
| `wait_stage` | Block until the run lands an output or errors — no polling. Also polls a headless job |
| `queue_status` | Status, progress, and `/view` outputs of one headless job. Does not block |
| `list_queue` | Headless jobs (queued, running, done, error). Not the remote-machine `jobs` tool |
| `resume_queue` | After checking `/queue` and `/history/{prompt_id}`, requeue a job ComfyUI has no record of. Startup does not do this unless `headless-auto-resume` is on (default off) |
| `cancel_stage` | Stop an in-flight page run. Headless: drop a job that has not been submitted; a prompt already in ComfyUI is left running |
| `remove_stage` | Delete a node |
| `arrange_canvas` | Tidy the canvas layout (rows, columns, grids) |

**Native ComfyUI graph**

| Tool | What it does |
|---|---|
| `graph_get` | Read the native canvas graph — every node, widget, link and group, ComfyTV or not |
| `graph_edit` | Batch-edit the native graph: add/remove/clone nodes, set widgets, wire links, group, collapse, bypass, pack/unpack subgraphs — one undo step |
| `graph_run` | Queue the native graph. With a page open, waits up to `wait_s`. With no page, pass `prompt` (API format) or `workflow` (UI format) and get a `job_id` back immediately |
| `workflow_create` | Author a brand-new workflow file for a stage kind from an API-format graph |
| `canvas_command` | Whitelisted native commands: undo/redo, save, fit view, interrupt, refresh node definitions |
| `canvas_focus` | Select a node and glide the user's viewport to it — show the user what changed |

**See & judge**

| Tool | What it does |
|---|---|
| `view_image` | Returns the actual image (downscaled) so the agent can look at it |
| `fx_preview` | Render a ~1.2s window of a video through one FX stage's real filter chain and see its middle frame — iterate look params cheaply before a full render |
| `media_probe` | Video duration/fps/resolution/audio |
| `media_frame` | Extract a frame from a video as a PNG URL |
| `media_waveform` | Render an audio waveform image |
| `media_timeline` | Composite filmstrip + waveform view of a video range, silence gaps detected as cut candidates |
| `pick_output` | Choose among multi-image output candidates |

**Configure**

| Tool | What it does |
|---|---|
| `workflow_get` / `workflow_edit` | Read a workflow's input bindings and node inventory; bind/unbind stage values to workflow inputs, validated against the real graph |
| `stage_params` | Define custom stage parameters, then bind them as `option:<key>` |
| `asset_edit` | Save outputs into the asset library, rename, categorize |
| `series_library` | Read and update a series folder's `library.yaml` (stable `ID@vN#view` codes, lock, new version, resolve, JSON manifest import) |

`resolve` (a code, a list, or shot mode) returns the view written on the code. Omit `#view` and a character, look, or outfit uses `#front` when that file exists. It does not also attach `#headshot`. series-kit `compile` mounts `#headshot` for characters in addition to `#front`; pass `CHR_…@vN#headshot` to `resolve` when you want that file. Shot mode resolves only the codes already stored on the shot, including STY, LOC, and PRP.

**Rich editors**

| Tool | What it does |
|---|---|
| `scene_get/edit/capture/record` | Drive a Scene 3D stage: characters, lights, cameras, animation clips, camera paths and the multi-shot director timeline (add_shot/patch_shot/set_path) |
| `director_get/director_edit` | Read and edit the Director clip timeline |
| `storyboard` | Read, replace, or validate a Storyboard stage's full shot document (`shots.json` fields and custom keys). `get_stage` only previews a large `storyboard_data` value. `validate` sums `duration_s` for the episode total (`cut_to` is only checked against that shot). The `target_duration_s` comparison is skipped when the board cannot reach the target even at 15s per shot (a single shot or a short partial extract) |
| `script_writer` | Script sessions under the project `script/` directory: versions, selection rewrite, and sha256 manifest import. Does not write `library.yaml`. `to_shots` is not included |

## Patterns that matter

**The production loop.** `stage_catalog` → `add_stage` (prompt, workflow, `asset_refs`) → `connect_stages` → `run_stage` → `wait_stage` → `outputs` → `view_image` to QC → iterate.

**Waiting without polling.** `wait_stage` blocks server-side (default 25 s per call, max 170 s) and returns the new output the second it lands. On timeout it returns `after_output_id` — call again with it to keep waiting. Total wait is unbounded; long renders just take a few re-calls.

**Mentions are 1-based positions.** Every stage keeps one ordered media list per type — wired inputs and `asset_refs` together, in the order `get_stage` reports under `media`. Prompt tokens like `@image_1` / `@video_1` address that list by position (`@image_1` is the first image). On `VideoStage`, `DirectorStage` and `ShotImagesStage` that same image order is `<Picture 1>`..`<Picture 9>`: `get_stage` adds `picture` on each image entry, and a video workflow that accepts fewer than 9 images errors instead of dropping the extras. `VideoStage` and `ShotImagesStage` store the seed on the `seed` widget (`widgets.seed`; `-1` means unset). A seed still sitting in `custom_params` is read once and written onto that widget. Removing or reordering media renumbers the tokens in the stage's prompt for you; `set_stage` `media_order` reorders a list, and `connect_stages` reports the position of the connection it made. Out-of-range tokens expand to nothing — and the tools warn you when that happens.

**Real vision.** `view_image` is the only tool that returns pixels. For video QC: `media_frame` to pull a frame, then `view_image` to actually look at it. Never judge an image by its filename.

**Binding a freshly linked workflow.** `workflow_get` shows the API graph's node inventory; `workflow_edit` binds stage values with sources like `option:seed`, `computed:width` / `computed:height` (sizing engine), `main_prompt`, `upstream_image:value[0]` or `literal:...`. If the workflow needs a parameter ComfyTV has no key for, create one with `stage_params` first and bind `option:<its key>`. `reset_to_preset` is your undo.

**Multi-machine.** `servers` lists configured machines with live load; `set_stage {server: <id>}` routes a stage's runs there. Results land back on the local machine.

**Two graph layers.** Stage tools (`add_stage`/`set_stage`/…) work at the ComfyTV product layer; the `graph_*` family edits the underlying native ComfyUI graph — any node from any plugin. Use stages when one exists for the job; drop to `graph_edit` + `graph_run` to build raw pipelines, and `canvas_focus` to show the user what you changed.

**Skills.** When the user has [Agent Skills](skills.md) installed, the `skill` tool's description lists them; read a matching skill before acting and follow it. Each enabled skill is also exposed as an **MCP prompt** — in Claude Code they appear as `/mcp__comfytv__<skill-name>` slash commands for explicit invocation.

## Headless queue

With no ComfyTV page open:

1. `add_stage` / `set_stage` / `connect_stages` / `graph_edit` edit a graph stored per project in the ComfyTV database (`comfytv_headless_graphs`). `graph_get` reads it back. An open, freshly mirroring page is still the source of truth for that project — close it (or let the mirror go stale) before driving the headless graph.
2. `run_stage` and `graph_run` insert a row in `comfytv_headless_jobs` and return `{job_id, status: "queued"}` without waiting. `graph_run` accepts an API-format `prompt` (what you would POST to `/prompt`) or a UI-format `workflow`. Omit both to run the stored graph.
3. A worker submits **one** job to ComfyUI's in-process prompt queue — the same queue `GET /queue` reads — and only when that queue has nothing running and nothing pending. ComfyTV never calls `/interrupt` or `/free`.
4. Poll `queue_status` (or `list_queue`) for `queued` / `running` / `done` / `error`, plus `progress` and `outputs`. `created_at` / `updated_at` / `started_at` / `finished_at` are UTC ISO-8601 with an offset (`+00:00`). `started_at` is when the job is handed to ComfyUI; `finished_at` is set on done, error, or cancel. `completed_nodes` is filled from execution progress, and on success lists every node in the prompt. There is no fixed timeout to wait out.
5. Jobs survive a process restart. Startup checks `GET /queue` (running and pending) and `GET /history/{prompt_id}` before doing anything else. A prompt still in the queue is re-attached and watched. A prompt already in history has its outputs collected and is not submitted again. A prompt ComfyUI has no record of is **not** submitted again unless you call `resume_queue` or turn on the `headless-auto-resume` setting (default off). That resubmit drops stages whose outputs were already saved and rewires downstream inputs to those outputs. Passing an `error` job id to `resume_queue` queues that job again only after the same queue and history check.

The new tables are created by the existing startup `create_all`. Nothing in this path binds a port; ComfyUI stays on `127.0.0.1`.

### Verify on ComfyUI 0.45.0 with no browser

1. Start ComfyUI v0.45.0 (commit `989d93d`) with ComfyTV enabled and MCP on. Do not open the ComfyTV page.
2. Confirm `server_info` reports `write_tools_need_open_tab: false` and `headless_queue: "one_at_a_time"`.
3. `graph_run` with a small API `prompt` (or `run_stage` after `add_stage`). The call returns a `job_id` at once.
4. `GET http://127.0.0.1:8188/queue` shows at most one ComfyTV prompt. A second `graph_run` stays `queued` in `list_queue` until the first leaves `/queue`.
5. `queue_status` with that `job_id` moves to `done` and lists `/view` URLs, including for a run longer than a few minutes (no client timeout to extend).
6. Restart ComfyUI while a job is unfinished. After it is back, `queue_status` shows `interrupted` (`headless-auto-resume` is off). `resume_queue` resubmits only when `/queue` and `/history/{prompt_id}` have no record of that prompt, and only the stages that did not save an output. A prompt still running stays attached and nothing new is submitted.

## Requirements and behavior notes

- When a page is open and its mirror is fresh, write tools and runs still go to that page. The mirror activates lazily after the first MCP call (retry `get_canvas` after ~10 s on a fresh connection).
- With no page, use the headless queue above. `queue_status` / `list_queue` are the headless jobs; `jobs` is still the remote-machine list.
- MCP clients cache `tools/list` per session — reconnect after the server gains new tools.
- If your client enforces a per-call tool timeout, poll `queue_status` instead of blocking inside `graph_run`. `wait_stage` still slices long waits (`timeout_s`, default 90, max 170) and can be re-called.

## See also

- [ComfyTV Bot](bot.md) — the embedded chat agent built on these same tools
- [Agent Skills](skills.md) — instruction packs served through the `skill` tool and MCP prompts
- [Custom workflows](custom-workflows.md) — what bindings are, hand-edited
- [Sidebar](sidebar.md) — the Settings panel with the MCP switch
