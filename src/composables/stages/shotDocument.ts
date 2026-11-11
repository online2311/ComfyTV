/**
 * Drama-pipeline shot documents and the legacy 16-column view.
 *
 * A production shots.json is stored as-is. `extras` is the crowd-count
 * integer from that file, not a custom-field bag. Keys outside the standard
 * set stay on the shot. Asset codes (CHR_/LOOK_/LOC_/PRP_/STY_/VOX_, NEW:,
 * episode:) are not resolved.
 */

export const DOCUMENT_FIELDS = [
  'series', 'episode', 'title_zh', 'fps', 'target_duration_s', 'new_assets', 'shots',
] as const

export const SHOT_FIELDS = [
  'shot_id', 'scene_id', 'order', 'beats', 'duration_s', 'cut_to', 'frames',
  'mode', 'shot_size', 'angle', 'move', 'lens', 'cam_idx',
  'purpose', 'reveal', 'characters', 'extras', 'location', 'props',
  'start_state', 'unique_action', 'end_state', 'variation_type',
  'dialogue', 'continuity', 'first_frame_desc', 'last_frame_desc', 'motion_desc',
  'lighting', 'audio', 'refs', 'prompts', 'video_plan',
] as const

export const LEGACY_FIELDS = [
  'shot_no', 'duration', 'scene_purpose', 'character', 'character_desc',
  'character_img', 'reference_img', 'action', 'emotion', 'scene_tags', 'sfx',
  'prompt', 'image_prompt', 'motion_prompt', 'image_url', 'reference_images', 'seed',
] as const

/** Scalars the main card does not already edit, shown in the fields panel. */
export const PANEL_FIELDS = [
  'shot_id', 'scene_id', 'mode', 'variation_type',
  'angle', 'move', 'lens', 'cam_idx',
  'reveal', 'start_state', 'end_state',
  'first_frame_desc', 'last_frame_desc',
  'cut_to', 'frames', 'extras',
] as const

/** Nested standard objects the panel shows as JSON. */
export const JSON_FIELDS = ['beats', 'continuity', 'audio', 'video_plan'] as const

const STANDARD_KEYS = new Set<string>([...SHOT_FIELDS, ...LEGACY_FIELDS])

export interface ShotView {
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
}

export function isProductionShot(shot: Record<string, unknown> | null | undefined): boolean {
  if (!shot) return false
  if (typeof shot.shot_id === 'string') return true
  if ('duration_s' in shot) return true
  return Array.isArray(shot.dialogue)
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

export function imagePromptOf(raw: Record<string, unknown>): string {
  const prompts = raw.prompts
  if (isRecord(prompts)) {
    if (typeof prompts.h3_image === 'string' && prompts.h3_image) return prompts.h3_image
    if (typeof prompts.krea2 === 'string') return prompts.krea2
  }
  return String(raw.image_prompt ?? raw.prompt ?? '')
}

export function separateKreaPrompt(raw: Record<string, unknown>): string | null {
  const prompts = raw.prompts
  if (!isRecord(prompts)) return null
  if (typeof prompts.h3_image !== 'string' || typeof prompts.krea2 !== 'string') return null
  return prompts.krea2
}

function durationOf(raw: Record<string, unknown>): number {
  if (typeof raw.duration_s === 'number' && Number.isFinite(raw.duration_s)) return raw.duration_s
  const n = parseInt(String(raw.duration ?? raw.duration_s ?? 3), 10)
  return Math.max(1, n || 3)
}

function characterLabel(raw: Record<string, unknown>): string {
  if (Array.isArray(raw.characters)) {
    return raw.characters
      .map(row => (isRecord(row) ? String(row.id ?? '') : ''))
      .filter(Boolean)
      .join('、')
  }
  return String(raw.character ?? '')
}

function emotionOf(raw: Record<string, unknown>): string {
  if (typeof raw.emotion === 'string' && raw.emotion) return raw.emotion
  if (Array.isArray(raw.dialogue)) {
    const parts = raw.dialogue
      .map(row => (isRecord(row) ? String(row.emotion ?? '') : ''))
      .filter(Boolean)
    if (parts.length) return parts.join('、')
  }
  return String(raw.emotion ?? '')
}

function sceneTagsOf(raw: Record<string, unknown>): string {
  if (typeof raw.scene_tags === 'string' && raw.scene_tags) return raw.scene_tags
  if (isRecord(raw.location)) {
    return [raw.location.id, raw.location.camera_card, raw.location.lighting_state]
      .filter(value => typeof value === 'string' && value)
      .join(', ')
  }
  return String(raw.scene_tags ?? '')
}

function sfxOf(raw: Record<string, unknown>): string {
  if (typeof raw.sfx === 'string' && raw.sfx) return raw.sfx
  if (isRecord(raw.audio)) {
    const ambience = typeof raw.audio.ambience === 'string' ? raw.audio.ambience : ''
    const effects = Array.isArray(raw.audio.sfx)
      ? raw.audio.sfx.map(item => {
        if (typeof item === 'string') return item
        return isRecord(item) ? String(item.desc ?? '') : ''
      })
      : []
    const joined = [ambience, ...effects].filter(Boolean).join(', ')
    if (joined) return joined
  }
  return String(raw.sfx ?? '')
}

export function dialogueText(raw: Record<string, unknown>): string {
  if (Array.isArray(raw.dialogue)) {
    return raw.dialogue.map(row => {
      if (!isRecord(row)) return ''
      const speaker = String(row.speaker ?? '')
      const text = String(row.text ?? '')
      return speaker ? `${speaker}: ${text}` : text
    }).filter(Boolean).join('\n')
  }
  return String(raw.dialogue ?? '')
}

export function viewFromSource(raw: Record<string, unknown>, index = 0, refs: string[] = []): ShotView {
  const imagePrompt = imagePromptOf(raw)
  return {
    shot_no: String(raw.shot_no ?? raw.order ?? index + 1),
    duration: durationOf(raw),
    image_url: refs[0] ?? null,
    prompt: imagePrompt,
    image_prompt: imagePrompt,
    scene_purpose: String(raw.purpose ?? raw.scene_purpose ?? ''),
    character: characterLabel(raw),
    character_desc: String(raw.character_desc ?? ''),
    shot_size: String(raw.shot_size ?? ''),
    action: String(raw.unique_action ?? raw.action ?? ''),
    emotion: emotionOf(raw),
    scene_tags: sceneTagsOf(raw),
    lighting: String(raw.lighting ?? ''),
    sfx: sfxOf(raw),
    dialogue: dialogueText(raw),
    motion_prompt: String(raw.motion_desc ?? raw.motion_prompt ?? ''),
    reference_images: refs,
    seed: null,
  }
}

function writeImagePrompt(source: Record<string, unknown>, value: string): void {
  if (isRecord(source.prompts)) {
    const prompts = source.prompts
    if (typeof prompts.h3_image === 'string' || !('krea2' in prompts)) prompts.h3_image = value
    else prompts.krea2 = value
    return
  }
  source.image_prompt = value
  source.prompt = value
}

export function applyEditorField(source: Record<string, unknown>, key: string, value: unknown): void {
  if (key === 'scene_purpose') {
    if ('purpose' in source || (isProductionShot(source) && !('scene_purpose' in source))) source.purpose = value
    else source.scene_purpose = value
    return
  }
  if (key === 'image_prompt' || key === 'prompt') {
    writeImagePrompt(source, String(value ?? ''))
    return
  }
  if (key === 'action') {
    if ('unique_action' in source || (isProductionShot(source) && !('action' in source))) source.unique_action = value
    else source.action = value
    return
  }
  if (key === 'motion_prompt') {
    if ('motion_desc' in source || (isProductionShot(source) && !('motion_prompt' in source))) source.motion_desc = value
    else source.motion_prompt = value
    return
  }
  if (key === 'duration') {
    const n = parseInt(String(value), 10)
    if (!Number.isFinite(n)) return
    if ('duration_s' in source || (isProductionShot(source) && !('duration' in source))) source.duration_s = n
    else source.duration = n
    return
  }
  if (key === 'dialogue' && Array.isArray(source.dialogue)) return
  if (key === 'character' && Array.isArray(source.characters)) return
  source[key] = value
}

export function setPromptKey(source: Record<string, unknown>, key: string, value: string): void {
  if (!isRecord(source.prompts)) source.prompts = {}
  ;(source.prompts as Record<string, unknown>)[key] = value
}

export function customKeys(source: Record<string, unknown>): string[] {
  return Object.keys(source).filter(key => !STANDARD_KEYS.has(key) && key !== 'id')
}

export function panelKeys(source: Record<string, unknown>): string[] {
  return PANEL_FIELDS.filter(key => key in source && !isComplex(source[key]))
}

export function jsonKeys(source: Record<string, unknown>): string[] {
  return JSON_FIELDS.filter(key => key in source)
}

export function isComplex(value: unknown): boolean {
  return Array.isArray(value) || isRecord(value)
}

export function coerceScalar(previous: unknown, text: string): unknown {
  if (typeof previous === 'number') {
    const n = Number(text)
    return Number.isFinite(n) ? n : previous
  }
  if (typeof previous === 'boolean') return text === 'true'
  if (previous === null) return text === '' ? null : text
  return text
}

export function objectRefs(source: Record<string, unknown>): Record<string, unknown>[] | null {
  if (!Array.isArray(source.refs)) return null
  if (!source.refs.every(row => isRecord(row))) return null
  return source.refs as Record<string, unknown>[]
}

export function dialogueRows(source: Record<string, unknown>): Record<string, unknown>[] | null {
  if (!Array.isArray(source.dialogue)) return null
  if (!source.dialogue.every(row => isRecord(row))) return null
  return source.dialogue as Record<string, unknown>[]
}

export function characterRows(source: Record<string, unknown>): Record<string, unknown>[] | null {
  if (!Array.isArray(source.characters)) return null
  if (!source.characters.every(row => isRecord(row))) return null
  return source.characters as Record<string, unknown>[]
}

export function blankProductionSource(order: number): Record<string, unknown> {
  return {
    shot_id: '',
    scene_id: '',
    order,
    duration_s: 6,
    mode: 'Ref2VA',
    shot_size: '',
    angle: '',
    move: '',
    purpose: '',
    characters: [],
    extras: 0,
    location: { id: '', camera_card: null, lighting_state: null },
    props: [],
    dialogue: [],
    refs: [],
    prompts: { krea2: '', h3_image: '' },
    continuity: { prev: null, next: null, cut_motive: '' },
    variation_type: 'small',
  }
}
