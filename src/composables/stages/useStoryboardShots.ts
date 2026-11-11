import { ref, toRaw, watch } from 'vue'

import { t } from '@/i18n'

import { app, type LGraphNode } from '@/lib/comfyApp'
import { useStageStore, type StageState } from '@/stores/stageStore'
import { uploadBlob } from '@/utils/uploadCanvas'
import { bindWidgetCallback, readWidgetStr, writeWidget } from '@/utils/widget'
import {
  applyEditorField,
  blankProductionSource,
  characterRows,
  coerceScalar,
  customKeys,
  dialogueRows,
  isProductionShot,
  objectRefs,
  setPromptKey,
  viewFromSource,
} from '@/composables/stages/shotDocument'

export interface Shot {
  id: string
  shot_no: string
  duration: number
  image_url: string | null
  prompt: string
  image_prompt: string
  scene_purpose: string
  character: string
  character_desc: string
  shot_size: string
  action: string
  emotion: string
  scene_tags: string
  lighting: string
  sfx: string
  dialogue: string
  motion_prompt: string
  reference_images: string[]
  seed: number | null
  /** Stored shot. The fields above are a view; serialize starts from this. */
  source: Record<string, unknown>
  [k: string]: unknown
}

export const MAX_SHOT_REFS = 9
const LEGACY_WRITE_KEYS = [
  'shot_no', 'duration', 'prompt', 'image_prompt', 'scene_purpose', 'character',
  'character_desc', 'shot_size', 'action', 'emotion', 'scene_tags', 'lighting',
  'sfx', 'dialogue', 'motion_prompt', 'reference_images', 'image_url', 'seed',
] as const

function asRefUrl(value: unknown): string | null {
  if (value && typeof value === 'object') {
    const row = value as Record<string, unknown>
    return asRefUrl(row.image_url) || asRefUrl(row.url)
      || asRefUrl(row.payload_url) || asRefUrl(row.src)
  }
  if (typeof value !== 'string') return null
  const text = value.trim()
  if (!text) return null
  const lower = text.toLowerCase()
  if (
    lower.startsWith('http://') || lower.startsWith('https://')
    || lower.startsWith('/view') || lower.startsWith('data:image/')
    || lower.includes('filename=')
  ) return text
  return null
}

function urlsFrom(raw: unknown): string[] {
  if (typeof raw === 'string') {
    const one = asRefUrl(raw)
    return one ? [one] : []
  }
  if (!Array.isArray(raw)) return []
  const out: string[] = []
  for (const item of raw) {
    const url = asRefUrl(item)
    if (url && !out.includes(url)) out.push(url)
  }
  return out
}

export function referenceImagesOf(raw: Record<string, unknown>): string[] {
  const listed = urlsFrom(raw.reference_images)
  if (listed.length) return listed
  const refs = urlsFrom(raw.refs)
  if (refs.length) return refs
  const one = asRefUrl(raw.image_url) || asRefUrl(raw.reference_img)
  return one ? [one] : []
}

const WHOLE_SEED = /^[+-]?\d+$/

export function seedOf(raw: unknown): number | null {
  if (typeof raw === 'number') return Number.isSafeInteger(raw) ? raw : null
  if (typeof raw !== 'string' || !WHOLE_SEED.test(raw.trim())) return null
  const n = Number(raw.trim())
  return Number.isSafeInteger(n) ? n : null
}

function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

function cloneSource(source: Record<string, unknown>): Record<string, unknown> {
  return structuredClone(toRaw(source))
}

function blankLegacySource(no: number): Record<string, unknown> {
  return {
    shot_no: String(no), duration: 3, prompt: '', image_prompt: '',
    scene_purpose: '', character: '', character_desc: '', shot_size: '',
    action: '', emotion: '', scene_tags: '', lighting: '', sfx: '',
    dialogue: '', motion_prompt: '', reference_images: [], image_url: null,
    seed: null,
  }
}

function shotFromSource(source: Record<string, unknown>, index: number): Shot {
  const refs = referenceImagesOf(source)
  return {
    id: newId(),
    source,
    ...viewFromSource(source, index, refs),
    seed: seedOf(source.seed),
  }
}

function project(shot: Shot, index: number) {
  const refs = referenceImagesOf(shot.source)
  const view = viewFromSource(shot.source, index, refs)
  Object.assign(shot, view, { seed: seedOf(shot.source.seed) })
}

export function useStoryboardShots(node: LGraphNode, state: StageState) {
  const store = useStageStore()

  const shots = ref<Shot[]>([])
  const uploadingId = ref<string | null>(null)
  const regeneratingId = ref<string | null>(null)
  const fileInputEl = ref<HTMLInputElement | null>(null)
  const importInputEl = ref<HTMLInputElement | null>(null)
  const customDraft = ref<Record<string, { key: string, value: string }>>({})
  let envelope: Record<string, unknown> = {}
  let pendingUploadShotId: string | null = null
  let pendingAppendRef = false
  let lastWritten = ''
  let ignoreExternalOutput = false
  let pendingRunResult = false

  function boardIsProduction(): boolean {
    return Boolean(envelope.episode || envelope.series || envelope.target_duration_s)
      || shots.value.some(shot => isProductionShot(shot.source))
  }

  function serializeShot(shot: Shot, index: number): Record<string, unknown> {
    if (isProductionShot(shot.source)) {
      const out = cloneSource(shot.source)
      if ('order' in out) out.order = index + 1
      return out
    }
    const out = cloneSource(shot.source)
    const view: Record<string, unknown> = {
      shot_no: String(index + 1),
      duration: shot.duration,
      prompt: shot.prompt,
      image_prompt: shot.image_prompt,
      scene_purpose: shot.scene_purpose,
      character: shot.character,
      character_desc: shot.character_desc,
      shot_size: shot.shot_size,
      action: shot.action,
      emotion: shot.emotion,
      scene_tags: shot.scene_tags,
      lighting: shot.lighting,
      sfx: shot.sfx,
      dialogue: shot.dialogue,
      motion_prompt: shot.motion_prompt,
      reference_images: shot.reference_images,
      image_url: shot.reference_images[0] ?? shot.image_url ?? null,
      seed: shot.seed,
    }
    for (const key of LEGACY_WRITE_KEYS) out[key] = view[key]
    return out
  }

  function serialize(): string {
    return JSON.stringify({
      ...envelope,
      shots: shots.value.map(serializeShot),
    })
  }

  function commit() {
    const json = serialize()
    lastWritten = json
    writeWidget(node, 'storyboard_data', json)
    store.applyExecutedPayload(state, { output: [json] })
  }

  function loadFromJson(raw: string): boolean {
    if (!raw) return false
    try {
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed?.shots)) return false
      const nextEnvelope: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(parsed)) {
        if (key !== 'shots') nextEnvelope[key] = value
      }
      envelope = nextEnvelope
      shots.value = parsed.shots.map((shot: unknown, index: number) => {
        const source = (shot && typeof shot === 'object')
          ? structuredClone(shot) as Record<string, unknown>
          : {}
        return shotFromSource(source, index)
      })
      return true
    } catch {
      return false
    }
  }

  function restore() {
    const raw = readWidgetStr(node, 'storyboard_data', '')
    if (loadFromJson(raw)) { lastWritten = raw; return }
    if (state.output && loadFromJson(String(state.output))) {
      lastWritten = String(state.output)
    }
  }

  function blankShot(no: number): Shot {
    const source = boardIsProduction() ? blankProductionSource(no) : blankLegacySource(no)
    return shotFromSource(source, no - 1)
  }

  function addShot() {
    shots.value.push(blankShot(shots.value.length + 1))
    commit()
  }
  function removeShot(id: string) {
    shots.value = shots.value.filter(shot => shot.id !== id)
    commit()
  }
  function move(idx: number, dir: -1 | 1) {
    const j = idx + dir
    if (j < 0 || j >= shots.value.length) return
    const arr = shots.value
    ;[arr[idx], arr[j]] = [arr[j], arr[idx]]
    commit()
  }
  function findShot(id: string): Shot | undefined {
    return shots.value.find(shot => shot.id === id)
  }
  function setField<K extends keyof Shot>(
    id: string, key: K, value: Shot[K], mirror?: keyof Shot,
  ) {
    const shot = findShot(id); if (!shot) return
    applyEditorField(shot.source, String(key), value)
    if (mirror && mirror !== key) applyEditorField(shot.source, String(mirror), value)
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setDuration(id: string, v: number) {
    const shot = findShot(id); if (!shot) return
    const n = Math.max(1, Math.min(60, Math.round(v)))
    applyEditorField(shot.source, 'duration', n)
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function writeRefs(shot: Shot) {
    if (isProductionShot(shot.source)) {
      if (shot.reference_images.length) shot.source.reference_images = [...shot.reference_images]
      else delete shot.source.reference_images
      if (shot.image_url) shot.source.image_url = shot.image_url
      else delete shot.source.image_url
      return
    }
    shot.source.reference_images = [...shot.reference_images]
    shot.source.image_url = shot.image_url
  }
  function syncPrimary(shot: Shot) {
    shot.image_url = shot.reference_images[0] ?? null
    writeRefs(shot)
  }
  function setImage(id: string, url: string | null) {
    const shot = findShot(id); if (!shot) return
    const tail = shot.reference_images.slice(1)
    const next = url
      ? [url, ...tail.filter(item => item !== url)]
      : tail
    if (next.length > MAX_SHOT_REFS && next.length > shot.reference_images.length) return
    shot.reference_images = next
    syncPrimary(shot)
    commit()
  }
  function addRef(id: string, url: string) {
    const shot = findShot(id); if (!shot) return
    if (!url || shot.reference_images.includes(url) || shot.reference_images.length >= MAX_SHOT_REFS) return
    shot.reference_images = [...shot.reference_images, url]
    syncPrimary(shot)
    commit()
  }
  function removeRef(id: string, index: number) {
    const shot = findShot(id); if (!shot) return
    shot.reference_images = shot.reference_images.filter((_, i) => i !== index)
    syncPrimary(shot)
    commit()
  }
  function setSeed(id: string, raw: string) {
    const shot = findShot(id); if (!shot) return
    shot.seed = seedOf(raw.trim() === '' ? null : raw)
    if (isProductionShot(shot.source)) {
      if (shot.seed === null) delete shot.source.seed
      else shot.source.seed = shot.seed
    } else {
      shot.source.seed = shot.seed
    }
    commit()
  }

  function setSourceScalar(id: string, key: string, text: string) {
    const shot = findShot(id); if (!shot || !(key in shot.source)) return
    shot.source[key] = coerceScalar(shot.source[key], text)
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setSourceJson(id: string, key: string, text: string) {
    const shot = findShot(id); if (!shot) return
    try {
      shot.source[key] = JSON.parse(text)
    } catch {
      return
    }
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setDialogueField(id: string, index: number, key: string, value: string) {
    const shot = findShot(id); if (!shot) return
    const rows = dialogueRows(shot.source)
    const line = rows?.[index]
    if (!line) return
    line[key] = value
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setAssetRef(id: string, index: number, key: 'role' | 'src', value: string) {
    const shot = findShot(id); if (!shot) return
    const rows = objectRefs(shot.source)
    const row = rows?.[index]
    if (!row) return
    row[key] = value
    commit()
  }
  function setCharacterId(id: string, index: number, value: string) {
    const shot = findShot(id); if (!shot) return
    const rows = characterRows(shot.source)
    const row = rows?.[index]
    if (!row) return
    row.id = value
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setLocationId(id: string, value: string) {
    const shot = findShot(id); if (!shot || !shot.source.location || typeof shot.source.location !== 'object') return
    ;(shot.source.location as Record<string, unknown>).id = value
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function setPropId(id: string, index: number, value: string) {
    const shot = findShot(id); if (!shot || !Array.isArray(shot.source.props)) return
    const row = shot.source.props[index]
    if (!row || typeof row !== 'object') return
    ;(row as Record<string, unknown>).id = value
    commit()
  }
  function setKreaPrompt(id: string, value: string) {
    const shot = findShot(id); if (!shot) return
    setPromptKey(shot.source, 'krea2', value)
    project(shot, shots.value.indexOf(shot))
    commit()
  }
  function draftFor(id: string) {
    if (!customDraft.value[id]) customDraft.value[id] = { key: '', value: '' }
    return customDraft.value[id]
  }
  function setDraft(id: string, field: 'key' | 'value', value: string) {
    draftFor(id)[field] = value
  }
  function addCustomField(id: string) {
    const shot = findShot(id); if (!shot) return
    const draft = draftFor(id)
    const key = draft.key.trim()
    if (!key || key in shot.source) return
    shot.source[key] = draft.value
    draft.key = ''
    draft.value = ''
    commit()
  }
  function removeCustomField(id: string, key: string) {
    const shot = findShot(id); if (!shot) return
    if (!customKeys(shot.source).includes(key)) return
    delete shot.source[key]
    commit()
  }
  function setCustomScalar(id: string, key: string, text: string) {
    const shot = findShot(id); if (!shot || !customKeys(shot.source).includes(key)) return
    shot.source[key] = coerceScalar(shot.source[key], text)
    commit()
  }

  function importDocument(raw: string): boolean {
    if (!loadFromJson(raw)) return false
    commit()
    return true
  }
  function pickImport() {
    importInputEl.value?.click()
  }
  async function onImportPicked(e: Event) {
    const input = e.target as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    const text = await file.text()
    if (importDocument(text)) return
    ;(app as any)?.extensionManager?.toast?.add?.({
      severity: 'warn',
      summary: t('storyboard.importFailed'),
      life: 5000,
    })
  }

  async function regenerateShot(shotId: string, targetNo: number) {
    if (regeneratingId.value) return
    const idx = shots.value.findIndex(shot => shot.id === shotId)
    if (idx < 0) return
    regeneratingId.value = shotId
    try {
      const workflow   = readWidgetStr(node, 'workflow', '')
      const premise    = readWidgetStr(node, 'main_prompt', '')
      const characters = readWidgetStr(node, 'characters', '')
      const body = {
        workflow,
        premise,
        characters,
        shots: shots.value.map(serializeShot),
        target_shot_no: targetNo,
      }
      const resp = await (app as any).api.fetchApi('/comfytv/storyboard/regenerate_shot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (resp.status !== 200) {
        const text = await resp.text()
        throw new Error(`${resp.status} ${text.slice(0, 200)}`)
      }
      const data = await resp.json() as { shot: Partial<Shot> }
      if (!data?.shot) throw new Error('no shot in response')
      const current = shots.value[idx]
      const keep = {
        image_url: current.image_url,
        reference_images: [...current.reference_images],
        seed: current.seed,
      }
      for (const [key, value] of Object.entries(data.shot)) {
        if (key === 'image_url' || key === 'reference_images' || key === 'seed' || key === 'id' || key === 'source') continue
        applyEditorField(current.source, key, value)
      }
      project(current, idx)
      current.image_url = keep.image_url
      current.reference_images = keep.reference_images
      current.seed = keep.seed
      current.shot_no = String(targetNo)
      if (!isProductionShot(current.source)) current.source.shot_no = String(targetNo)
      commit()
    } catch (err: any) {
      console.error('[ComfyTV/storyboard] regenerate failed', err)
      ;(app as any)?.extensionManager?.toast?.add?.({
        severity: 'warn',
        summary: t('storyboard.regenerateFailed'),
        detail: String(err?.message || err),
        life: 5000,
      })
    } finally {
      regeneratingId.value = null
    }
  }

  function pickFile(shotId: string) {
    pendingUploadShotId = shotId
    pendingAppendRef = false
    fileInputEl.value?.click()
  }
  function pickExtra(shotId: string) {
    pendingUploadShotId = shotId
    pendingAppendRef = true
    fileInputEl.value?.click()
  }
  async function onFilePicked(e: Event) {
    const input = e.target as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    const shotId = pendingUploadShotId
    const append = pendingAppendRef
    pendingUploadShotId = null
    pendingAppendRef = false
    if (!file || !shotId) return

    uploadingId.value = shotId
    try {
      const url = await uploadBlob(file, { subfolder: 'comfytv/storyboard', filename: file.name })
      if (append) addRef(shotId, url)
      else setImage(shotId, url)
    } catch (err) {
      console.error('[ComfyTV/storyboard] ref upload failed', err)
    } finally {
      uploadingId.value = null
    }
  }

  restore()
  ignoreExternalOutput = lastWritten !== ''
  bindWidgetCallback(node, 'storyboard_data', (value) => {
    const text = String(value ?? '')
    if (!text || text === lastWritten) return
    if (!loadFromJson(text)) return
    const json = serialize()
    if (json === text) {
      lastWritten = text
    } else {
      lastWritten = json
      writeWidget(node, 'storyboard_data', json)
    }
    ignoreExternalOutput = true
    store.applyExecutedPayload(state, { output: [lastWritten] })
  })
  if (node) {
    const origOnConfigure = node.onConfigure
    node.onConfigure = function (info: any) {
      origOnConfigure?.call(this, info)
      restore()
      ignoreExternalOutput = lastWritten !== ''
    }
  }

  watch(() => state.running, (running, was) => {
    if (was && !running) pendingRunResult = true
  })

  watch(() => state.output, (out) => {
    if (!out || out === lastWritten) return
    if (pendingRunResult) {
      pendingRunResult = false
      ignoreExternalOutput = false
    } else if (ignoreExternalOutput && lastWritten) {
      store.setOutputSlot(state, 0, lastWritten)
      return
    }
    if (loadFromJson(String(out))) {
      lastWritten = String(out)
      writeWidget(node, 'storyboard_data', out)
    }
  })

  return {
    shots, uploadingId, regeneratingId, fileInputEl, importInputEl, customDraft,
    addShot, removeShot, move,
    setField, setDuration, setImage, addRef, removeRef, setSeed,
    setSourceScalar, setSourceJson, setDialogueField, setAssetRef, setCharacterId,
    setLocationId, setPropId, setKreaPrompt, setDraft, addCustomField, removeCustomField,
    setCustomScalar, importDocument, pickImport, onImportPicked,
    regenerateShot,
    pickFile, pickExtra, onFilePicked,
  }
}
