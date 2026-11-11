import { parseParamItems, serializeParamItems } from '@/composables/stages/useCustomParams'
import type { LGraphNode } from '@/lib/comfyApp'
import { getWidget, readWidgetNum, readWidgetStr, writeWidget } from '@/utils/widget'

/** Widget value meaning "no stage seed"; a custom_params seed may still apply. */
export const UNSET_SEED = -1

const LEGACY_PICTURE = /^picture_\d+$/
const LIST_KEYS = new Set(['reference_images', 'pictures', 'refs'])

export function coerceSeed(raw: unknown): number | null {
  if (typeof raw === 'boolean' || raw == null || raw === '') return null
  if (typeof raw === 'number' && Number.isInteger(raw)) return raw
  if (typeof raw === 'string' && /^[+-]?\d+$/.test(raw.trim())) return Number(raw.trim())
  return null
}

export function migrateSeedParams(
  seed: number,
  customParams: string,
  hasImages: boolean,
): { seed: number; customParams: string; changed: boolean } {
  const items = parseParamItems(customParams)
  let nextSeed = Number.isFinite(seed) ? seed : UNSET_SEED
  if (nextSeed === UNSET_SEED) {
    const legacy = coerceSeed(items.find(item => item.key === 'seed')?.value)
    if (legacy != null) nextSeed = legacy
  }
  const kept = items.filter(item => {
    if (item.key === 'seed' && nextSeed !== UNSET_SEED) return false
    if (hasImages && (LEGACY_PICTURE.test(item.key) || LIST_KEYS.has(item.key))) return false
    return true
  })
  const custom = serializeParamItems(kept)
  const changed = nextSeed !== seed || custom !== serializeParamItems(items)
  return { seed: nextSeed, customParams: custom, changed }
}

export function migrateNodeSeed(node: LGraphNode, hasImages = false): void {
  if (!getWidget(node, 'seed')) return
  const current = readWidgetNum(node, 'seed', UNSET_SEED)
  const raw = readWidgetStr(node, 'custom_params', '{}')
  const next = migrateSeedParams(current, raw, hasImages)
  if (!next.changed) return
  writeWidget(node, 'seed', next.seed)
  writeWidget(node, 'custom_params', next.customParams)
}
