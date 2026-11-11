import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'

import { app } from '@/lib/comfyApp'
import type { StageState } from '@/stores/stageStore'

import { seedOf, useStoryboardShots } from './useStoryboardShots'

const jsonResp = (data: any, status = 200) =>
  new Response(JSON.stringify(data), {
    status, headers: { 'content-type': 'application/json' },
  })

function makeWidget(name: string, value: any = '') {
  return { name, value, callback: vi.fn() }
}

function makeNode(widgets: any[] = []): any {
  return { widgets, onConfigure: null as any }
}

function makeState(over: Partial<StageState> = {}): StageState {
  return reactive({
    kind: 'storyboard',
    variant: 'generator',
    outputType: 'COMFYTV_STORYBOARD',
    output: null,
    outputs: [null],
    running: false,
    inputs: [],
    mainPrompt: '',
    ...over,
  }) as StageState
}

describe('useStoryboardShots', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('add/remove/move maintain order', () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const state = makeState()
    const { shots, addShot, removeShot, move } = useStoryboardShots(node, state)
    addShot(); addShot(); addShot()
    expect(shots.value).toHaveLength(3)

    const [a, b, c] = shots.value.map(s => s.id)
    move(0, 1)
    expect(shots.value.map(s => s.id)).toEqual([b, a, c])
    removeShot(a)
    expect(shots.value.map(s => s.id)).toEqual([b, c])
  })

  it('setField writes the field and the optional mirror', () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const state = makeState()
    const { shots, addShot, setField } = useStoryboardShots(node, state)
    addShot()
    const id = shots.value[0].id
    setField(id, 'image_prompt', 'a cat', 'prompt')
    expect(shots.value[0].image_prompt).toBe('a cat')
    expect(shots.value[0].prompt).toBe('a cat')
  })

  it('setDuration clamps to [1, 60]', () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const { shots, addShot, setDuration } = useStoryboardShots(node, makeState())
    addShot()
    const id = shots.value[0].id
    setDuration(id, 0)
    expect(shots.value[0].duration).toBe(1)
    setDuration(id, 999)
    expect(shots.value[0].duration).toBe(60)
    setDuration(id, 5.7)
    expect(shots.value[0].duration).toBe(6)
  })

  it('commit writes serialized JSON to the storyboard_data widget and state.output', () => {
    const w = makeWidget('storyboard_data', '')
    const onWrite = w.callback
    const node = makeNode([w])
    const state = makeState()
    const { addShot } = useStoryboardShots(node, state)
    addShot()
    expect(w.value).not.toBe('')
    expect(onWrite).toHaveBeenCalled()
    const parsed = JSON.parse(w.value)
    expect(Array.isArray(parsed.shots)).toBe(true)
    expect(parsed.shots[0].shot_no).toBe('1')
    expect(state.output).toBe(w.value)
  })

  it('restore loads shots from the widget value on mount', () => {
    const seed = {
      shots: [
        { shot_no: '1', duration: 4, image_prompt: 'first', character: 'A' },
        { shot_no: '2', duration: 7, prompt: 'fallback prompt' },
      ],
    }
    const node = makeNode([makeWidget('storyboard_data', JSON.stringify(seed))])
    const { shots } = useStoryboardShots(node, makeState())
    expect(shots.value).toHaveLength(2)
    expect(shots.value[0].duration).toBe(4)
    expect(shots.value[0].image_prompt).toBe('first')
    expect(shots.value[0].character).toBe('A')
    expect(shots.value[1].image_prompt).toBe('fallback prompt')
    expect(shots.value[1].prompt).toBe('fallback prompt')
  })

  it('restore picks up state.output when widget is empty', () => {
    const seed = { shots: [{ shot_no: '1', duration: 3, image_prompt: 'from-state' }] }
    const node = makeNode([makeWidget('storyboard_data', '')])
    const state = makeState({ output: JSON.stringify(seed) })
    const { shots } = useStoryboardShots(node, state)
    expect(shots.value).toHaveLength(1)
    expect(shots.value[0].image_prompt).toBe('from-state')
  })

  it('keeps a saved board when an older execution snapshot arrives', async () => {
    const saved = {
      shots: [{ shot_no: '1', image_prompt: 'kept', seed: 3,
                reference_images: ['/view?filename=kept.png&type=input'] }],
    }
    const node = makeNode([makeWidget('storyboard_data', JSON.stringify(saved))])
    const state = makeState()
    const { shots } = useStoryboardShots(node, state)
    state.output = JSON.stringify({ shots: [{ shot_no: '1', image_prompt: 'from db' }] })
    await nextTick()
    expect(shots.value[0].image_prompt).toBe('kept')
    expect(shots.value[0].seed).toBe(3)
    expect(state.output).toContain('kept.png')
  })

  it('adopts the storyboard produced by a finished run', async () => {
    const node = makeNode([makeWidget('storyboard_data', JSON.stringify({
      shots: [{ shot_no: '1', image_prompt: 'old' }],
    }))])
    const state = makeState()
    const { shots } = useStoryboardShots(node, state)
    state.running = true
    await nextTick()
    state.running = false
    state.output = JSON.stringify({ shots: [{ shot_no: '1', image_prompt: 'fresh' }] })
    await nextTick()
    expect(shots.value[0].image_prompt).toBe('fresh')
  })

  it('watch on state.output re-loads when an upstream payload arrives', async () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const state = makeState()
    const { shots } = useStoryboardShots(node, state)
    expect(shots.value).toHaveLength(0)

    state.output = JSON.stringify({
      shots: [{ shot_no: '1', duration: 3, image_prompt: 'pushed' }],
    })
    await nextTick()
    expect(shots.value).toHaveLength(1)
    expect(shots.value[0].image_prompt).toBe('pushed')
  })

  it('setImage sets the image_url and commits', () => {
    const w = makeWidget('storyboard_data', '')
    const node = makeNode([w])
    const { shots, addShot, setImage } = useStoryboardShots(node, makeState())
    addShot()
    setImage(shots.value[0].id, '/view?x=1')
    expect(shots.value[0].image_url).toBe('/view?x=1')
    expect(shots.value[0].reference_images).toEqual(['/view?x=1'])
    const parsed = JSON.parse(w.value)
    expect(parsed.shots[0].image_url).toBe('/view?x=1')
    expect(parsed.shots[0].reference_images).toEqual(['/view?x=1'])
    expect(parsed.shots[0].seed).toBeNull()
  })

  it('does not turn a partial seed into an override the backend would ignore', () => {
    for (const seed of ['12.5', '12oops', [12], true, 12.5, '']) {
      const saved = {
        shots: [{ shot_no: '1', image_prompt: 'x', seed }],
      }
      const w = makeWidget('storyboard_data', JSON.stringify(saved))
      const { shots, setField } = useStoryboardShots(makeNode([w]), makeState())
      expect(shots.value[0].seed).toBeNull()
      setField(shots.value[0].id, 'scene_purpose', 'edited')
      expect(JSON.parse(w.value).shots[0].seed).toBeNull()
    }
    expect(seedOf('42')).toBe(42)
    expect(seedOf(0)).toBe(0)
    expect(seedOf(' -7 ')).toBe(-7)
  })

  it('round-trips per-shot refs and seed, and ignores a non-url reference', () => {
    const seed = {
      shots: [{
        shot_no: '1',
        image_prompt: 'door',
        reference_images: ['/view?filename=a.png&type=input', 'not a url'],
        seed: 42,
        reference_img: '特写',
      }],
    }
    const w = makeWidget('storyboard_data', JSON.stringify(seed))
    const node = makeNode([w])
    const { shots, addRef, setSeed, removeRef } = useStoryboardShots(node, makeState())
    expect(shots.value[0].reference_images).toEqual(['/view?filename=a.png&type=input'])
    expect(shots.value[0].seed).toBe(42)
    addRef(shots.value[0].id, 'https://example.com/b.png')
    setSeed(shots.value[0].id, '')
    const parsed = JSON.parse(w.value)
    expect(parsed.shots[0].reference_images).toEqual([
      '/view?filename=a.png&type=input',
      'https://example.com/b.png',
    ])
    expect(parsed.shots[0].seed).toBeNull()
    removeRef(shots.value[0].id, 0)
    expect(shots.value[0].image_url).toBe('https://example.com/b.png')
  })

  it('regenerateShot merges the backend response into the shot at the target index', async () => {
    const node = makeNode([
      makeWidget('storyboard_data', ''),
      makeWidget('workflow', 'qwen3'),
      makeWidget('main_prompt', 'a story'),
      makeWidget('characters', 'A, B'),
    ])
    const { shots, addShot, regenerateShot } = useStoryboardShots(node, makeState())
    addShot()
    const id = shots.value[0].id
    const originalImage = '/keep-this-image'
    shots.value[0].image_url = originalImage

    const fetchApi = (app as any).api.fetchApi as ReturnType<typeof vi.fn>
    fetchApi.mockResolvedValueOnce(jsonResp({
      shot: {
        duration: 5,
        image_prompt: 'a regenerated scene',
        character: 'A',
        image_url: '/server-suggested-but-ignored',
      },
    }))

    await regenerateShot(id, 1)
    expect(shots.value[0].id).toBe(id)
    expect(shots.value[0].duration).toBe(5)
    expect(shots.value[0].image_prompt).toBe('a regenerated scene')
    expect(shots.value[0].character).toBe('A')
    expect(shots.value[0].image_url).toBe(originalImage)

    const [path, init] = fetchApi.mock.calls[0]
    expect(path).toBe('/comfytv/storyboard/regenerate_shot')
    const body = JSON.parse(init.body as string)
    expect(body.workflow).toBe('qwen3')
    expect(body.premise).toBe('a story')
    expect(body.characters).toBe('A, B')
    expect(body.target_shot_no).toBe(1)
  })

  it('regenerateShot serializes: regeneratingId stays pinned to the in-flight shot', async () => {
    const node = makeNode([
      makeWidget('storyboard_data', ''), makeWidget('workflow', 'x'),
      makeWidget('main_prompt', ''), makeWidget('characters', ''),
    ])
    const { shots, addShot, regenerateShot, regeneratingId } = useStoryboardShots(node, makeState())
    addShot(); addShot()
    const fetchApi = (app as any).api.fetchApi as ReturnType<typeof vi.fn>

    let resolve!: (r: Response) => void
    fetchApi.mockReturnValueOnce(new Promise<Response>(r => { resolve = r }))

    const id1 = shots.value[0].id
    const id2 = shots.value[1].id
    const inflight = regenerateShot(id1, 1)
    expect(regeneratingId.value).toBe(id1)

    await regenerateShot(id2, 2)
    expect(regeneratingId.value).toBe(id1)

    resolve(jsonResp({ shot: { duration: 2 } }))
    await inflight
    expect(regeneratingId.value).toBeNull()
  })

  it('regenerateShot reports backend HTTP failure via toast and resets the lock', async () => {
    const toastAdd = vi.fn()
    ;(app as any).extensionManager = { toast: { add: toastAdd } }
    const node = makeNode([
      makeWidget('storyboard_data', ''), makeWidget('workflow', 'x'),
      makeWidget('main_prompt', ''), makeWidget('characters', ''),
    ])
    const { shots, addShot, regenerateShot, regeneratingId } = useStoryboardShots(node, makeState())
    addShot()
    const fetchApi = (app as any).api.fetchApi as ReturnType<typeof vi.fn>
    fetchApi.mockResolvedValueOnce(new Response('boom', { status: 500 }))

    await regenerateShot(shots.value[0].id, 1)
    expect(toastAdd).toHaveBeenCalled()
    expect(toastAdd.mock.calls[0][0].severity).toBe('warn')
    expect(regeneratingId.value).toBeNull()
  })

  it('pickFile + onFilePicked uploads the file and writes /view URL into image_url', async () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const { shots, addShot, pickFile, fileInputEl, onFilePicked } = useStoryboardShots(node, makeState())
    addShot()
    const id = shots.value[0].id
    fileInputEl.value = { click: vi.fn() } as any
    pickFile(id)
    expect(fileInputEl.value?.click).toHaveBeenCalled()

    const fetchApi = (app as any).api.fetchApi as ReturnType<typeof vi.fn>
    fetchApi.mockResolvedValueOnce(jsonResp({ name: 'uploaded.png' }))

    const file = new File(['data'], 'src.png', { type: 'image/png' })
    const fakeEvent = { target: { files: [file], value: 'src.png' } } as unknown as Event
    await onFilePicked(fakeEvent)
    expect(shots.value[0].image_url).toBe(
      '/view?filename=uploaded.png&subfolder=comfytv%2Fstoryboard&type=input',
    )
  })

  it('keeps a production shots.json, including custom keys, across an edit', () => {
    const doc = {
      series: 'gumu-sx',
      episode: 'E01',
      title_zh: '玉女心经',
      shots: [{
        shot_id: 'E01S01_01',
        scene_id: 'E01S01',
        order: 1,
        duration_s: 6,
        purpose: 'intro',
        characters: [{ id: 'CHR_xiaolongnv@v1', visible: 'face' }],
        extras: 0,
        dialogue: [{
          line_id: 'L001', speaker: 'CHR_xiaolongnv', kind: 'onscreen',
          text: '你好', emotion: '平静',
        }],
        refs: [{ role: 'first_frame', src: 'episode:E01S01_01/first.png' }],
        prompts: { krea2: 'krea', h3_image: 'h3' },
        first_frame_desc: 'candle',
        last_frame_desc: null,
        angle: '平视',
        custom_note: { keep: true },
      }],
    }
    const w = makeWidget('storyboard_data', JSON.stringify(doc))
    const { shots, setField, setDialogueField, importDocument } = useStoryboardShots(makeNode([w]), makeState())
    expect(shots.value[0].scene_purpose).toBe('intro')
    expect(shots.value[0].image_prompt).toBe('h3')
    expect(shots.value[0].dialogue).toContain('CHR_xiaolongnv: 你好')
    expect(shots.value[0].character).toContain('CHR_xiaolongnv@v1')
    setField(shots.value[0].id, 'scene_purpose', 'revised')
    setDialogueField(shots.value[0].id, 0, 'text', '改过')
    const parsed = JSON.parse(w.value)
    expect(parsed.series).toBe('gumu-sx')
    expect(parsed.title_zh).toBe('玉女心经')
    const shot = parsed.shots[0]
    expect(shot.purpose).toBe('revised')
    expect(shot.scene_purpose).toBeUndefined()
    expect(shot.prompt).toBeUndefined()
    expect(shot.seed).toBeUndefined()
    expect(shot.shot_no).toBeUndefined()
    expect(shot.extras).toBe(0)
    expect(shot.custom_note).toEqual({ keep: true })
    expect(shot.dialogue[0].text).toBe('改过')
    expect(shot.dialogue[0].speaker).toBe('CHR_xiaolongnv')
    expect(shot.dialogue[0].line_id).toBe('L001')
    expect(shot.refs).toEqual(doc.shots[0].refs)
    expect(shot.prompts).toEqual({ krea2: 'krea', h3_image: 'h3' })
    expect(shot.last_frame_desc).toBeNull()

    const again = structuredClone(parsed)
    again.shots[0].purpose = 'from file'
    expect(importDocument(JSON.stringify(again))).toBe(true)
    expect(JSON.parse(w.value)).toEqual(again)
  })

  it('keeps a legacy custom key when an editor field changes', () => {
    const saved = {
      shots: [{ shot_no: '1', duration: 3, image_prompt: 'a', pipeline_note: 'keep' }],
    }
    const w = makeWidget('storyboard_data', JSON.stringify(saved))
    const { shots, setField } = useStoryboardShots(makeNode([w]), makeState())
    setField(shots.value[0].id, 'scene_purpose', 'edited')
    const shot = JSON.parse(w.value).shots[0]
    expect(shot.pipeline_note).toBe('keep')
    expect(shot.scene_purpose).toBe('edited')
    expect(shot.shot_no).toBe('1')
  })

  it('reloads a document written into the widget without dropping fields', () => {
    const w = makeWidget('storyboard_data', '')
    const { shots } = useStoryboardShots(makeNode([w]), makeState())
    const doc = {
      episode: 'E01',
      shots: [{ shot_id: 'E01S01_01', duration_s: 6, purpose: 'from mcp', camera_notes: 'wide' }],
    }
    w.value = JSON.stringify(doc)
    w.callback?.(w.value)
    expect(shots.value[0].scene_purpose).toBe('from mcp')
    expect(shots.value[0].source.camera_notes).toBe('wide')
    expect(JSON.parse(String(w.value)).shots[0].camera_notes).toBe('wide')
  })

  it('onFilePicked is a no-op if no shot is pending (e.g. user picked without click)', async () => {
    const node = makeNode([makeWidget('storyboard_data', '')])
    const { shots, addShot, onFilePicked } = useStoryboardShots(node, makeState())
    addShot()
    const fetchApi = (app as any).api.fetchApi as ReturnType<typeof vi.fn>
    fetchApi.mockClear()
    const file = new File(['data'], 'x.png', { type: 'image/png' })
    const fakeEvent = { target: { files: [file], value: 'x.png' } } as unknown as Event
    await onFilePicked(fakeEvent)
    expect(fetchApi).not.toHaveBeenCalled()
    expect(shots.value[0].image_url).toBeNull()
  })
})
