import { describe, expect, it } from 'vitest'

import { migrateSeedParams, UNSET_SEED } from './seedField'

describe('migrateSeedParams', () => {
  it('copies a legacy custom_params seed onto the field and writes it out', () => {
    const out = migrateSeedParams(
      UNSET_SEED,
      JSON.stringify({ items: [
        { key: 'seed', value: 20261113 },
        { key: 'negative_prompt', value: 'blur' },
      ] }),
      false,
    )
    expect(out.changed).toBe(true)
    expect(out.seed).toBe(20261113)
    expect(JSON.parse(out.customParams).items).toEqual([
      { key: 'negative_prompt', value: 'blur' },
    ])
  })

  it('keeps a set seed field and drops the old custom_params seed', () => {
    const out = migrateSeedParams(
      11,
      JSON.stringify({ items: [{ key: 'seed', value: 7 }] }),
      false,
    )
    expect(out.seed).toBe(11)
    expect(JSON.parse(out.customParams).items).toEqual([])
  })

  it('keeps legacy picture slots until images are wired, then drops them', () => {
    const raw = JSON.stringify({ items: [
      { key: 'picture_1', value: '/view?filename=a.png&type=input' },
      { key: 'picture_2', value: '/view?filename=b.png&type=input' },
      { key: 'picture_3', value: '/view?filename=c.png&type=input' },
    ] })
    expect(migrateSeedParams(UNSET_SEED, raw, false).customParams).toBe(
      JSON.stringify({ items: JSON.parse(raw).items }),
    )
    const wired = migrateSeedParams(UNSET_SEED, raw, true)
    expect(JSON.parse(wired.customParams).items).toEqual([])
  })

  it('treats 0 as a real seed', () => {
    const out = migrateSeedParams(0, JSON.stringify({ items: [{ key: 'seed', value: 9 }] }), false)
    expect(out.seed).toBe(0)
    expect(JSON.parse(out.customParams).items).toEqual([])
  })
})
