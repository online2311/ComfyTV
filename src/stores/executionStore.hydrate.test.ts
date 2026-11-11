import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

vi.mock('@/api', () => ({
  listRemoteJobs: vi.fn(async () => ({ jobs: [] })),
}))

vi.mock('@/i18n', () => ({
  t: (key: string) => key,
}))

import { listRemoteJobs } from '@/api'
import { useExecutionStore } from './executionStore'

describe('executionStore remote hydration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(listRemoteJobs).mockReset()
  })

  it('keeps cancel-requested remote jobs attached after a reload', async () => {
    vi.mocked(listRemoteJobs).mockImplementation(async (status?: string) => ({
      jobs: [{
        id: status === 'cancel-requested' ? 'c1' : 'r1',
        server_label: 'spark',
        project_id: 'p',
        stage_node_id: status === 'cancel-requested' ? '8' : '4',
        status: status ?? 'running',
      }],
    }))
    const store = useExecutionStore()
    expect(await store.remoteJobForNode('8')).toBe('c1')
    expect(store.remoteJobs.get('4')).toBe('r1')
    expect(listRemoteJobs).toHaveBeenCalledWith('running')
    expect(listRemoteJobs).toHaveBeenCalledWith('cancel-requested')
  })
})
