<template>
  <div class="ctv:flex ctv:flex-col ctv:gap-1.5 ctv:size-full">
    <div class="ctv:flex ctv:items-center ctv:gap-2">
      <span class="ctv:text-[11px] ctv:font-semibold ctv:text-muted-foreground">{{ $t('storyboard.shots') }} · {{ shots.length }}</span>
      <button
        type="button"
        class="ctv:ml-auto ctv:py-0.5 ctv:px-2.5 ctv:text-[11px] ctv:rounded ctv:cursor-pointer
               ctv:bg-secondary-background ctv:border ctv:border-border-subtle ctv:text-base-foreground"
        @click="pickImport"
      >{{ $t('storyboard.importJson') }}</button>
      <button
        type="button"
        class="ctv:py-0.5 ctv:px-2.5 ctv:text-[11px] ctv:rounded ctv:cursor-pointer
               ctv:bg-primary-background/15 ctv:border ctv:border-primary-background/40 ctv:text-primary-background
               ctv:hover:bg-primary-background/25"
        @click="addShot"
      >+ {{ $t('storyboard.addShot') }}</button>
    </div>

    <div v-if="shots.length === 0" class="ctv:text-[11px] ctv:text-muted-foreground/60 ctv:p-3 ctv:text-center">
      {{ $t('storyboard.empty') }}
    </div>

    <div v-else class="ctv:flex ctv:flex-col ctv:gap-1.5">
      <div
        v-for="(shot, idx) in shots"
        :key="shot.id"
        class="ctv:flex ctv:flex-col ctv:gap-1.5 ctv:p-2 ctv:rounded-md ctv:border ctv:border-border-subtle ctv:bg-base-foreground/5"
      >
        <header class="ctv:flex ctv:items-center ctv:gap-1.5 ctv:flex-wrap">
          <span class="ctv:text-[13px] ctv:font-bold ctv:text-base-foreground">#{{ idx + 1 }}</span>
          <label class="ctv:flex ctv:items-center ctv:gap-0.5 ctv:text-2xs ctv:text-muted-foreground">
            <input
              type="number" min="1" max="60" step="1"
              :value="shot.duration"
              class="ctv:w-[38px] ctv:py-0.5 ctv:px-1 ctv:rounded-sm ctv:text-[11px] ctv:font-mono
                     ctv:bg-secondary-background ctv:text-base-foreground ctv:border ctv:border-border-subtle"
              @change="(e) => setDuration(shot.id, Number((e.target as HTMLInputElement).value))"
            /><span>s</span>
          </label>
          <label class="ctv:flex ctv:items-center ctv:gap-0.5 ctv:text-2xs ctv:text-muted-foreground"
                 :title="$t('storyboard.seedHint')">
            <span>{{ $t('storyboard.seed') }}</span>
            <input
              type="number" step="1"
              :value="shot.seed ?? ''"
              :placeholder="$t('storyboard.seedAuto')"
              class="ctv:w-[72px] ctv:py-0.5 ctv:px-1 ctv:rounded-sm ctv:text-[11px] ctv:font-mono
                     ctv:bg-secondary-background ctv:text-base-foreground ctv:border ctv:border-border-subtle"
              @change="(e) => setSeed(shot.id, (e.target as HTMLInputElement).value)"
            />
          </label>
          <span v-if="shot.shot_size"
                class="ctv:py-px ctv:px-1.5 ctv:rounded-sm ctv:text-2xs ctv:bg-secondary-background ctv:text-base-foreground">{{ shot.shot_size }}</span>
          <span v-if="shot.character && shot.character !== '无'"
                class="ctv:py-px ctv:px-1.5 ctv:rounded-sm ctv:text-2xs ctv:bg-success-background/15 ctv:text-success-background">{{ shot.character }}</span>
          <div class="ctv:flex ctv:flex-col ctv:gap-px ctv:ml-auto">
            <button type="button" :class="moveBtn" :disabled="idx === 0"
                    @click="move(idx, -1)" :title="$t('storyboard.moveUp')"><i class="pi pi-chevron-up" /></button>
            <button type="button" :class="moveBtn" :disabled="idx === shots.length - 1"
                    @click="move(idx, 1)" :title="$t('storyboard.moveDown')"><i class="pi pi-chevron-down" /></button>
          </div>
          <button
            type="button"
            class="ctv:bg-transparent ctv:border-0 ctv:cursor-pointer ctv:text-xs ctv:px-0.5 ctv:opacity-70
                   ctv:hover:opacity-100 ctv:disabled:opacity-40 ctv:disabled:cursor-default"
            :disabled="regeneratingId === shot.id"
            :title="$t('storyboard.regenerate')"
            @click="regenerateShot(shot.id, idx + 1)"
          ><span v-if="regeneratingId === shot.id">…</span><i v-else class="pi pi-refresh" /></button>
          <button type="button"
                  class="ctv:bg-transparent ctv:border-0 ctv:cursor-pointer ctv:text-[13px]"
                  :title="$t('storyboard.remove')" @click="removeShot(shot.id)"><i class="pi pi-trash" /></button>
        </header>

        <textarea
          class="ctv:w-full ctv:box-border ctv:resize-none ctv:border-0 ctv:border-l-2 ctv:rounded-none
                 ctv:py-1 ctv:px-2 ctv:text-[11px] ctv:italic ctv:leading-snug ctv:min-h-[22px] ctv:[font-family:inherit]
                 ctv:bg-primary-background/5 ctv:border-primary-background/60 ctv:text-base-foreground
                 ctv:focus:outline ctv:focus:outline-1 ctv:focus:outline-primary-background/50 ctv:focus:bg-primary-background/10"
          :value="shot.scene_purpose"
          :placeholder="$t('storyboard.cols.scene_purpose')"
          rows="1"
          @input="(e) => setField(shot.id, 'scene_purpose', (e.target as HTMLTextAreaElement).value)"
        />

        <div class="ctv:flex ctv:gap-1.5">
          <div class="ctv-hover-host ctv:relative ctv:shrink-0 ctv:w-24 ctv:h-[72px] ctv:rounded ctv:overflow-hidden ctv:bg-black ctv:border ctv:border-border-subtle">
            <img v-if="shot.image_url" :src="shot.image_url" :alt="`shot ${idx + 1}`"
                 class="ctv:size-full ctv:object-cover" draggable="false" />
            <ViewFullButton
              v-if="shot.image_url"
              class="ctv:top-0.5 ctv:left-0.5"
              :url="shot.image_url"
              :label="`shot ${idx + 1}`"
            />
            <div v-else class="ctv:size-full ctv:flex ctv:items-center ctv:justify-center ctv:text-3xs ctv:text-white/35 ctv:text-center ctv:px-1">
              {{ $t('storyboard.noRef') }}
            </div>
            <button
              type="button"
              class="ctv:absolute ctv:bottom-0.5 ctv:right-0.5 ctv:size-5 ctv:p-0 ctv:border-0 ctv:rounded ctv:cursor-pointer ctv:text-[11px]
                     ctv:bg-black/60 ctv:text-white ctv:disabled:opacity-60"
              :disabled="uploadingId === shot.id"
              :title="$t('storyboard.uploadRef')"
              @click="pickFile(shot.id)"
            ><span v-if="uploadingId === shot.id">…</span><i v-else class="pi pi-upload" /></button>
            <button
              v-if="shot.image_url"
              type="button"
              class="ctv:absolute ctv:top-0.5 ctv:right-0.5 ctv:size-5 ctv:p-0 ctv:border-0 ctv:rounded ctv:cursor-pointer ctv:text-[11px]
                     ctv:bg-destructive-background/70 ctv:text-white"
              :title="$t('storyboard.clearRef')"
              @click="setImage(shot.id, null)"
            ><i class="pi pi-times" /></button>
          </div>

          <textarea
            class="ctv:flex-1 ctv:min-h-14 ctv:resize-y ctv:box-border ctv:py-1 ctv:px-1.5 ctv:rounded ctv:text-[11px] ctv:leading-snug
                   ctv:bg-secondary-background ctv:text-base-foreground ctv:border ctv:border-border-subtle"
            :value="shot.image_prompt"
            :placeholder="$t('storyboard.promptPlaceholder')"
            @input="(e) => setField(shot.id, 'image_prompt', (e.target as HTMLTextAreaElement).value, /*mirror=*/'prompt')"
          />
        </div>

        <div class="ctv:flex ctv:items-center ctv:gap-1 ctv:flex-wrap">
          <span class="ctv:text-2xs ctv:opacity-60">{{ $t('storyboard.refsCount', { n: shot.reference_images.length }) }}</span>
          <button
            type="button"
            class="ctv:py-0.5 ctv:px-1.5 ctv:text-2xs ctv:rounded ctv:cursor-pointer
                   ctv:bg-secondary-background ctv:border ctv:border-border-subtle ctv:text-base-foreground
                   ctv:disabled:opacity-40 ctv:disabled:cursor-default"
            :disabled="shot.reference_images.length >= MAX_SHOT_REFS || uploadingId === shot.id"
            @click="pickExtra(shot.id)"
          >+ {{ $t('storyboard.addRef') }}</button>
          <button
            v-for="(url, ri) in shot.reference_images"
            :key="`${shot.id}-${ri}`"
            type="button"
            class="ctv:relative ctv:w-8 ctv:h-8 ctv:p-0 ctv:rounded ctv:overflow-hidden ctv:border ctv:border-border-subtle ctv:cursor-pointer"
            :title="`${$t('storyboard.pictureSlot', { n: ri + 1 })} — ${$t('storyboard.removeRef')}`"
            @click="removeRef(shot.id, ri)"
          >
            <img :src="url" alt="" class="ctv:size-full ctv:object-cover" draggable="false" />
          </button>
        </div>

        <dl class="ctv:grid ctv:grid-cols-[max-content_1fr] ctv:gap-x-2.5 ctv:gap-y-[3px] ctv:m-0 ctv:text-2xs ctv:items-start">
          <template v-for="field in metaFields(shot)" :key="field.key">
            <dt class="ctv:opacity-50 ctv:whitespace-nowrap ctv:pt-1">{{ $t(field.label) }}</dt>
            <dd class="ctv:m-0">
              <textarea
                v-if="field.multiline"
                :class="metaInput(true)"
                :value="String(shot[field.key] ?? '')"
                rows="1"
                @input="(e) => setField(shot.id, field.key, (e.target as HTMLTextAreaElement).value)"
              />
              <input
                v-else
                :class="metaInput(false)"
                type="text"
                :value="String(shot[field.key] ?? '')"
                @input="(e) => setField(shot.id, field.key, (e.target as HTMLInputElement).value)"
              />
            </dd>
          </template>
        </dl>

        <details class="ctv:text-2xs ctv:py-[3px] ctv:px-1.5 ctv:rounded ctv:border ctv:border-dashed ctv:border-border-subtle"
                 :open="!!shot.character_desc">
          <summary class="ctv:cursor-pointer ctv:opacity-75 ctv:select-none ctv:pb-1 ctv:hover:opacity-100">{{ $t('storyboard.cols.character_desc') }}</summary>
          <textarea
            :class="metaInput(true, true)"
            :value="shot.character_desc"
            rows="2"
            @input="(e) => setField(shot.id, 'character_desc', (e.target as HTMLTextAreaElement).value)"
          />
        </details>
        <label v-if="kreaPrompt(shot) !== null"
               class="ctv:flex ctv:flex-col ctv:gap-0.5 ctv:text-2xs">
          <span class="ctv:opacity-60">krea2</span>
          <textarea
            :class="metaInput(true, true)"
            :value="kreaPrompt(shot) ?? ''"
            rows="2"
            @input="(e) => setKreaPrompt(shot.id, (e.target as HTMLTextAreaElement).value)"
          />
        </label>

        <details class="ctv:text-2xs ctv:py-[3px] ctv:px-1.5 ctv:rounded ctv:border ctv:border-dashed ctv:border-border-subtle"
                 :open="!!shot.source.shot_id">
          <summary class="ctv:cursor-pointer ctv:opacity-75 ctv:select-none ctv:pb-1 ctv:hover:opacity-100">{{ $t('storyboard.fields') }}</summary>
          <div class="ctv:flex ctv:flex-col ctv:gap-1 ctv:pt-1">
            <label v-for="key in panelKeys(shot.source)" :key="key" class="ctv:grid ctv:grid-cols-[max-content_1fr] ctv:gap-1.5 ctv:items-center">
              <span class="ctv:opacity-60">{{ key }}</span>
              <input
                :class="metaInput(false)"
                type="text"
                :value="shot.source[key] == null ? '' : String(shot.source[key])"
                @change="(e) => setSourceScalar(shot.id, key, (e.target as HTMLInputElement).value)"
              />
            </label>

            <template v-if="dialogueRows(shot.source)">
              <span class="ctv:opacity-60">{{ $t('storyboard.dialogueLines') }}</span>
              <div v-for="(line, li) in dialogueRows(shot.source)" :key="`${shot.id}-line-${li}`"
                   class="ctv:grid ctv:grid-cols-[1fr_4.5rem] ctv:gap-1">
                <input :class="metaInput(false)" type="text" :value="String(line.speaker ?? '')"
                       :placeholder="$t('storyboard.speaker')"
                       @change="(e) => setDialogueField(shot.id, li, 'speaker', (e.target as HTMLInputElement).value)" />
                <input :class="metaInput(false)" type="text" :value="String(line.kind ?? '')"
                       placeholder="kind"
                       @change="(e) => setDialogueField(shot.id, li, 'kind', (e.target as HTMLInputElement).value)" />
                <textarea class="ctv:col-span-2" :class="metaInput(true)" rows="2" :value="String(line.text ?? '')"
                          @change="(e) => setDialogueField(shot.id, li, 'text', (e.target as HTMLTextAreaElement).value)" />
              </div>
            </template>

            <template v-if="characterRows(shot.source)">
              <span class="ctv:opacity-60">{{ $t('storyboard.assetRefs') }}</span>
              <label v-for="(ch, ci) in characterRows(shot.source)" :key="`${shot.id}-chr-${ci}`"
                     class="ctv:grid ctv:grid-cols-[max-content_1fr] ctv:gap-1.5 ctv:items-center">
                <span class="ctv:opacity-50">{{ ch.visible || 'character' }}</span>
                <input :class="metaInput(false)" type="text" :value="String(ch.id ?? '')"
                       @change="(e) => setCharacterId(shot.id, ci, (e.target as HTMLInputElement).value)" />
              </label>
              <label v-if="isRecord(shot.source.location)" class="ctv:grid ctv:grid-cols-[max-content_1fr] ctv:gap-1.5 ctv:items-center">
                <span class="ctv:opacity-50">location</span>
                <input :class="metaInput(false)" type="text" :value="String((shot.source.location as Record<string, unknown>).id ?? '')"
                       @change="(e) => setLocationId(shot.id, (e.target as HTMLInputElement).value)" />
              </label>
              <label v-for="(prop, pi) in propRows(shot)" :key="`${shot.id}-prp-${pi}`"
                     class="ctv:grid ctv:grid-cols-[max-content_1fr] ctv:gap-1.5 ctv:items-center">
                <span class="ctv:opacity-50">prop</span>
                <input :class="metaInput(false)" type="text" :value="String(prop.id ?? '')"
                       @change="(e) => setPropId(shot.id, pi, (e.target as HTMLInputElement).value)" />
              </label>
            </template>

            <template v-if="objectRefs(shot.source)">
              <div v-for="(ref, ri) in objectRefs(shot.source)" :key="`${shot.id}-ref-${ri}`"
                   class="ctv:grid ctv:grid-cols-[5.5rem_1fr] ctv:gap-1">
                <input :class="metaInput(false)" type="text" :value="String(ref.role ?? '')"
                       @change="(e) => setAssetRef(shot.id, ri, 'role', (e.target as HTMLInputElement).value)" />
                <input :class="metaInput(false)" type="text" :value="String(ref.src ?? '')"
                       @change="(e) => setAssetRef(shot.id, ri, 'src', (e.target as HTMLInputElement).value)" />
              </div>
            </template>

            <template v-for="key in jsonKeys(shot.source)" :key="key">
              <span class="ctv:opacity-60">{{ key }}</span>
              <textarea :class="metaInput(true, true)" rows="3" :value="pretty(shot.source[key])"
                        @change="(e) => setSourceJson(shot.id, key, (e.target as HTMLTextAreaElement).value)" />
            </template>

            <template v-for="key in customKeys(shot.source)" :key="key">
              <div class="ctv:grid ctv:grid-cols-[1fr_auto] ctv:gap-1 ctv:items-start">
                <span class="ctv:opacity-60 ctv:pt-1">{{ key }}</span>
                <button type="button" class="ctv:bg-transparent ctv:border-0 ctv:cursor-pointer ctv:text-[11px]"
                        :title="$t('storyboard.removeField')" @click="removeCustomField(shot.id, key)">×</button>
                <textarea v-if="isComplex(shot.source[key])" class="ctv:col-span-2" :class="metaInput(true, true)" rows="2"
                          :value="pretty(shot.source[key])"
                          @change="(e) => setSourceJson(shot.id, key, (e.target as HTMLTextAreaElement).value)" />
                <input v-else class="ctv:col-span-2" :class="metaInput(false)" type="text"
                       :value="shot.source[key] == null ? '' : String(shot.source[key])"
                       @change="(e) => setCustomScalar(shot.id, key, (e.target as HTMLInputElement).value)" />
              </div>
            </template>

            <div class="ctv:grid ctv:grid-cols-[1fr_1fr_auto] ctv:gap-1">
              <input :class="metaInput(false)" type="text" :placeholder="$t('storyboard.fieldKey')"
                     :value="draft(shot.id).key"
                     @input="(e) => setDraft(shot.id, 'key', (e.target as HTMLInputElement).value)" />
              <input :class="metaInput(false)" type="text" :placeholder="$t('storyboard.fieldValue')"
                     :value="draft(shot.id).value"
                     @input="(e) => setDraft(shot.id, 'value', (e.target as HTMLInputElement).value)" />
              <button type="button" class="ctv:px-1.5 ctv:rounded ctv:cursor-pointer ctv:border ctv:border-border-subtle ctv:bg-secondary-background"
                      @click="addCustomField(shot.id)">+</button>
            </div>
          </div>
        </details>

        <details class="ctv:text-2xs ctv:py-[3px] ctv:px-1.5 ctv:rounded ctv:border ctv:border-dashed ctv:border-border-subtle"
                 :open="!!shot.motion_prompt">
          <summary class="ctv:cursor-pointer ctv:opacity-75 ctv:select-none ctv:pb-1 ctv:hover:opacity-100">{{ $t('storyboard.cols.motion_prompt') }}</summary>
          <textarea
            :class="metaInput(true, true)"
            :value="shot.motion_prompt"
            rows="2"
            @input="(e) => setField(shot.id, 'motion_prompt', (e.target as HTMLTextAreaElement).value)"
          />
        </details>
      </div>
    </div>

    <input
      ref="fileInputEl"
      type="file"
      accept="image/*"
      class="ctv:hidden"
      @change="onFilePicked"
    />
    <input
      ref="importInputEl"
      type="file"
      accept="application/json,.json"
      class="ctv:hidden"
      @change="onImportPicked"
    />

    <StageCard
      :state="state"
      :node="node"
      :on-run-request="onRunRequest"
      :on-cancel-request="onCancelRequest"
      :on-disconnect="onDisconnect"
      :on-action="onAction"
      hide-output
    />
  </div>
</template>

<script setup lang="ts">
import StageCard from '@/components/stages/StageCard.vue'
import ViewFullButton from '@/components/ViewFullButton.vue'
import { MAX_SHOT_REFS, useStoryboardShots, type Shot } from '@/composables/stages/useStoryboardShots'
import {
  characterRows,
  customKeys,
  dialogueRows,
  isComplex,
  isRecord,
  jsonKeys,
  objectRefs,
  panelKeys,
  separateKreaPrompt,
} from '@/composables/stages/shotDocument'
import type { LGraphNode } from '@/lib/comfyApp'
import type { StageState } from '@/stores/stageStore'

const META_FIELDS: ReadonlyArray<{ key: keyof Shot; label: string; multiline?: boolean }> = [
  { key: 'character',  label: 'storyboard.cols.character' },
  { key: 'shot_size',  label: 'storyboard.cols.shot_size' },
  { key: 'emotion',    label: 'storyboard.cols.emotion' },
  { key: 'scene_tags', label: 'storyboard.cols.scene_tags' },
  { key: 'lighting',   label: 'storyboard.cols.lighting' },
  { key: 'action',     label: 'storyboard.cols.action', multiline: true },
  { key: 'sfx',        label: 'storyboard.cols.sfx' },
  { key: 'dialogue',   label: 'storyboard.cols.dialogue' },
]

function metaFields(shot: Shot) {
  if (Array.isArray(shot.source?.dialogue)) {
    return META_FIELDS.filter(field => field.key !== 'dialogue' && field.key !== 'character')
  }
  if (Array.isArray(shot.source?.characters)) {
    return META_FIELDS.filter(field => field.key !== 'character')
  }
  return META_FIELDS
}

function kreaPrompt(shot: Shot): string | null {
  return separateKreaPrompt(shot.source)
}

function propRows(shot: Shot): Record<string, unknown>[] {
  const props = shot.source?.props
  if (!Array.isArray(props)) return []
  return props.filter(row => isRecord(row)) as Record<string, unknown>[]
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return ''
  }
}

function draft(id: string) {
  return customDraft.value[id] ?? { key: '', value: '' }
}

const props = defineProps<{
  state: StageState
  onRunRequest: () => void
  onCancelRequest: () => void
  onDisconnect: (slot: string) => void
  onAction: (id: string) => void
  node: LGraphNode
}>()

const {
  shots, uploadingId, regeneratingId, fileInputEl, importInputEl, customDraft,
  addShot, removeShot, move,
  setField, setDuration, setImage, removeRef, setSeed,
  setSourceScalar, setSourceJson, setDialogueField, setAssetRef, setCharacterId,
  setLocationId, setPropId, setKreaPrompt, setDraft, addCustomField, removeCustomField,
  setCustomScalar, pickImport, onImportPicked,
  regenerateShot,
  pickFile, pickExtra, onFilePicked,
} = useStoryboardShots(props.node, props.state)

const moveBtn = [
  'ctv:w-4 ctv:h-[13px] ctv:leading-none ctv:text-[8px] ctv:p-0 ctv:rounded-sm ctv:cursor-pointer',
  'ctv:bg-secondary-background ctv:text-muted-foreground ctv:border ctv:border-border-subtle',
  'ctv:disabled:opacity-30 ctv:disabled:cursor-default',
].join(' ')

const META_INPUT_BASE = 'ctv:w-full ctv:box-border ctv:py-0.5 ctv:px-1.5 ctv:text-2xs ctv:leading-snug ctv:rounded-sm ctv:[font-family:inherit]'
  + ' ctv:bg-secondary-background ctv:text-base-foreground ctv:border ctv:border-transparent'
  + ' ctv:hover:border-border-subtle'
  + ' ctv:focus:outline-none ctv:focus:border-primary-background/50 ctv:focus:bg-secondary-background-hover'
function metaInput(multiline: boolean, mono = false) {
  const ml = multiline ? ' ctv:min-h-[22px] ctv:resize-y' : ''
  const m = mono ? ' ctv:font-mono ctv:min-h-10' : ''
  return `${META_INPUT_BASE}${ml}${m}`
}
</script>
