import { ref, watch } from 'vue'

import { t } from '@/i18n'

import { app, type LGraphNode } from '@/lib/comfyApp'
import { useStageStore, type StageState } from '@/stores/stageStore'
import { uploadBlob } from '@/utils/uploadCanvas'
import { readWidgetStr, writeWidget } from '@/utils/widget'

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
  [k: string]: unknown
}

const MAX_SHOT_REFS = 8

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
    if (out.length >= MAX_SHOT_REFS) break
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

function blankShot(no: number): Shot {
  return {
    id: newId(), shot_no: String(no), duration: 3, image_url: null,
    prompt: '', image_prompt: '', scene_purpose: '', character: '',
    character_desc: '', shot_size: '', action: '', emotion: '',
    scene_tags: '', lighting: '', sfx: '', dialogue: '', motion_prompt: '',
    reference_images: [], seed: null,
  }
}

export function useStoryboardShots(node: LGraphNode, state: StageState) {
  const store = useStageStore()

  const shots = ref<Shot[]>([])
  const uploadingId = ref<string | null>(null)
  const regeneratingId = ref<string | null>(null)
  const fileInputEl = ref<HTMLInputElement | null>(null)
  let pendingUploadShotId: string | null = null
  let pendingAppendRef = false
  let lastWritten = ''
  let ignoreExternalOutput = false
  let pendingRunResult = false

  function serialize(): string {
    return JSON.stringify({
      shots: shots.value.map((s, i) => ({
        shot_no: String(i + 1),
        duration: s.duration,
        prompt: s.prompt,
        image_prompt: s.image_prompt,
        scene_purpose: s.scene_purpose,
        character: s.character,
        character_desc: s.character_desc,
        shot_size: s.shot_size,
        action: s.action,
        emotion: s.emotion,
        scene_tags: s.scene_tags,
        lighting: s.lighting,
        sfx: s.sfx,
        dialogue: s.dialogue,
        motion_prompt: s.motion_prompt,
        reference_images: s.reference_images,
        image_url: s.reference_images[0] ?? s.image_url ?? null,
        seed: s.seed,
      })),
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
      const p = JSON.parse(raw)
      if (!Array.isArray(p?.shots)) return false
      shots.value = p.shots.map((s: any, i: number) => {
        const dur = Math.max(1, parseInt(String(s.duration ?? 3), 10) || 3)
        const imgPrompt = String(s.image_prompt ?? s.prompt ?? '')
        const reference_images = referenceImagesOf(s)
        return {
          id: newId(),
          shot_no: String(s.shot_no ?? i + 1),
          duration: dur,
          image_url: reference_images[0] ?? null,
          reference_images,
          seed: seedOf(s.seed),
          prompt: imgPrompt,
          image_prompt: imgPrompt,
          scene_purpose: String(s.scene_purpose ?? ''),
          character: String(s.character ?? ''),
          character_desc: String(s.character_desc ?? ''),
          shot_size: String(s.shot_size ?? ''),
          action: String(s.action ?? ''),
          emotion: String(s.emotion ?? ''),
          scene_tags: String(s.scene_tags ?? ''),
          lighting: String(s.lighting ?? ''),
          sfx: String(s.sfx ?? ''),
          dialogue: String(s.dialogue ?? ''),
          motion_prompt: String(s.motion_prompt ?? ''),
        }
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

  function addShot() {
    shots.value.push(blankShot(shots.value.length + 1))
    commit()
  }
  function removeShot(id: string) {
    shots.value = shots.value.filter(s => s.id !== id)
    commit()
  }
  function move(idx: number, dir: -1 | 1) {
    const j = idx + dir
    if (j < 0 || j >= shots.value.length) return
    const arr = shots.value
    ;[arr[idx], arr[j]] = [arr[j], arr[idx]]
    commit()
  }
  function setField<K extends keyof Shot>(
    id: string, key: K, value: Shot[K], mirror?: keyof Shot,
  ) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    ;(s as any)[key] = value
    if (mirror) (s as any)[mirror] = value
    commit()
  }
  function setDuration(id: string, v: number) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    s.duration = Math.max(1, Math.min(60, Math.round(v))); commit()
  }
  function syncPrimary(s: Shot) {
    s.image_url = s.reference_images[0] ?? null
  }
  function setImage(id: string, url: string | null) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    const tail = s.reference_images.slice(1)
    s.reference_images = url
      ? [url, ...tail.filter(item => item !== url)].slice(0, MAX_SHOT_REFS)
      : tail
    syncPrimary(s)
    commit()
  }
  function addRef(id: string, url: string) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    if (!url || s.reference_images.includes(url) || s.reference_images.length >= MAX_SHOT_REFS) return
    s.reference_images = [...s.reference_images, url]
    syncPrimary(s)
    commit()
  }
  function removeRef(id: string, index: number) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    s.reference_images = s.reference_images.filter((_, i) => i !== index)
    syncPrimary(s)
    commit()
  }
  function setSeed(id: string, raw: string) {
    const s = shots.value.find(x => x.id === id); if (!s) return
    s.seed = seedOf(raw.trim() === '' ? null : raw)
    commit()
  }

  async function regenerateShot(shotId: string, targetNo: number) {
    if (regeneratingId.value) return
    const idx = shots.value.findIndex(s => s.id === shotId)
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
        shots: shots.value.map(s => ({ ...s, shot_no: s.shot_no })),
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
      const incoming = data.shot
      const merged: Shot = {
        ...current,
        ...incoming,
        id: current.id,
        image_url: current.image_url,
        reference_images: current.reference_images,
        seed: current.seed,
        shot_no: String(targetNo),
        duration: parseInt(String(incoming.duration ?? current.duration), 10) || current.duration,
        prompt: String(incoming.image_prompt ?? incoming.prompt ?? current.prompt ?? ''),
        image_prompt: String(incoming.image_prompt ?? incoming.prompt ?? current.image_prompt ?? ''),
      }
      shots.value[idx] = merged
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
    shots, uploadingId, regeneratingId, fileInputEl,
    addShot, removeShot, move,
    setField, setDuration, setImage, addRef, removeRef, setSeed,
    regenerateShot,
    pickFile, pickExtra, onFilePicked,
  }
}
