// free-model-aggregator-01: one server, two client APIs, ten free-tier providers.
//
// The server reads the trial brief (provider directory, quality table, workload manifest), probes
// every provider's catalog route in that provider's own wire, registers the catalog it built, then
// answers the client's twenty requests. Each request is routed to the highest-rated free model that
// meets the request's requirements and whose provider still has free-tier budget; when a provider
// refuses (402, 429 or 5xx) the request fails over to the next candidate instead of dying with it.
//
// Design notes:
//   - Both client wires (OpenAI Chat Completions, Anthropic Messages) are parsed into one neutral
//     conversation form, rendered into whichever provider wire serves the request (OpenAI,
//     Anthropic or Gemini), and the provider's answer is rendered back into the client's wire, as
//     JSON or as server-sent events. The echo-relevant fields (system text, last user text, tool
//     names, max_tokens, stop sequences, image bytes, JSON mode) survive every translation exactly,
//     because each wire's parser reads them the same way from the neutral form.
//   - Providers are always called non-streaming; a streamed client answer is synthesised from the
//     provider's whole answer, in the client wire's event order, with the provider's usage counts.
//   - Per-provider attempt caps sit at the smallest per-trial budget the public kit's provider table
//     allows, so budget exhaustion can never be the reason a call refuses; a provider that refuses
//     twice in a row is treated as out (an outage or a spent budget refuses the same way) and drops
//     to last-resort for the rest of the trial.
//   - Everything the server sends is a function of the brief, the requests and the providers'
//     answers: no wall clock (fixed timestamps in documents), no randomness.
//
// `node main.ts --list` prints the catalog document (the same object the register body carries) and
// exits 0; with no brief and no env it prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function strOr(v: unknown, fallback: string): string {
  return typeof v === 'string' ? v : fallback;
}

/** A stop list as the wires' parsers read it: a string is a one-item list, an array keeps its strings. */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text a content value carries: itself when a string, else its text blocks joined with '\n'. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents inside the prompt file, found by schema
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The provider's own path (/p/<id>) or an absolute base URL, as the directory spells it. */
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  provider: string;
  model: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  requires: Requirement;
}

interface Brief {
  providers: ProviderInfo[];
  ratings: Map<string, RatingRow>;
  manifest: ManifestEntry[];
}

/** Every balanced {...} span of `text` that parses as a JSON object carrying a string `schema`. */
function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep scanning.
    }
  }
  return out;
}

function readBrief(path: string): Brief {
  const brief: Brief = { providers: [], ratings: new Map(), manifest: [] };
  if (path === '') return brief;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    log(`cannot read the brief at ${path}: ${(err as Error).message}`);
    return brief;
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory !== undefined) {
    for (const raw of directory.providers as unknown[]) {
      if (!isRecord(raw)) continue;
      const id = strOr(raw.id, '');
      const wire = raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : raw.wire === 'openai' ? 'openai' : null;
      if (id === '' || wire === null) continue;
      const auth =
        raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
      const base =
        strOr(raw.path, '') !== ''
          ? strOr(raw.path, '')
          : strOr(raw.base_url, '') !== ''
            ? strOr(raw.base_url, '')
            : strOr(raw.base, '');
      brief.providers.push({
        id,
        name: strOr(raw.name, '') !== '' ? strOr(raw.name, '') : id,
        wire,
        auth,
        base,
        keyEnv: strOr(raw.key_env, ''),
        key: strOr(raw.key, ''),
      });
    }
  }
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  if (ratingsDoc !== undefined) {
    for (const raw of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(raw)) continue;
      const provider = strOr(raw.provider, '');
      const model = strOr(raw.model, '');
      if (provider === '' || model === '') continue;
      brief.ratings.set(`${provider}/${model}`, {
        provider,
        model,
        rating: numOrNull(raw.rating_milli) ?? 0,
        contextLength: numOrNull(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  if (workloadDoc !== undefined) {
    for (const raw of workloadDoc.manifest as unknown[]) {
      if (!isRecord(raw)) continue;
      const id = strOr(raw.id, '');
      if (id === '') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      brief.manifest.push({
        id,
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          minContext: numOrNull(req.min_context),
        },
      });
    }
  }
  return brief;
}

// ---------------------------------------------------------------------------
// Provider catalogs: probe each provider's own listing, in its own wire
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface ProviderState {
  info: ProviderInfo;
  /** The models the provider's own catalog route listed; empty until probed (or when it never answers). */
  listed: ListedModel[];
  probed: boolean;
  /** Inference calls this trial has sent the provider, refusals included. */
  attempts: number;
  /** Refusals since the provider last answered: two in a row reads as an outage or a spent budget. */
  refusalsInARow: number;
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const ANTHROPIC_VERSION = '2023-06-01';
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;

/** A provider path joined to the env's origin; an absolute base is used as it stands. */
function dialBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return ENV_URL;
  return `${ENV_URL}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme carries its key in; `none` (or no key) sends nothing. */
function authHeaders(info: ProviderInfo): Record<string, string> {
  if (info.key === '') return {};
  if (info.auth === 'bearer') return { authorization: `Bearer ${info.key}` };
  if (info.auth === 'x-api-key') return { 'x-api-key': info.key };
  if (info.auth === 'x-goog-api-key') return { 'x-goog-api-key': info.key };
  return {};
}

function listedModel(wire: Wire, raw: unknown): ListedModel | null {
  if (!isRecord(raw)) return null;
  const id = wire === 'gemini' ? strOr(raw.name, '').replace(/^models\//, '') : strOr(raw.id, '');
  if (id === '') return null;
  const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
  return {
    id,
    free: bool(raw.free),
    contextLength: numOrNull(raw.context_length) ?? numOrNull(raw.context_window) ?? numOrNull(raw.inputTokenLimit),
    tools: bool(raw.tools),
    vision: bool(raw.vision),
    json: bool(raw.json),
  };
}

/** Ask one provider for its catalog, in its wire; every failure leaves an empty, unprobed state. */
async function probe(state: ProviderState): Promise<void> {
  const info = state.info;
  const base = dialBase(info.base);
  const url = info.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(info) };
  if (info.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) {
      log(`probe of ${info.id} answered HTTP ${res.status}`);
      return;
    }
    const body: unknown = await res.json();
    if (!isRecord(body)) return;
    const list = info.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return;
    const seen = new Set<string>();
    for (const raw of list) {
      const model = listedModel(info.wire, raw);
      if (model === null || seen.has(model.id)) continue;
      seen.add(model.id);
      state.listed.push(model);
    }
    state.probed = true;
  } catch (err) {
    log(`probe of ${info.id} failed: ${(err as Error).message}`);
  }
}

// ---------------------------------------------------------------------------
// The catalog document: the same object --list prints and /register carries
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[] }[];
}

/** Whether a model is free: its listing's word first, the quality table's row behind it, else paid. */
function modelFree(brief: Brief, providerId: string, model: ListedModel): boolean {
  if (model.free !== null) return model.free;
  const row = brief.ratings.get(`${providerId}/${model.id}`);
  return row !== undefined ? row.free : false;
}

function buildCatalog(brief: Brief, states: ProviderState[]): Catalog {
  const providers: Catalog['providers'] = [];
  for (const state of states) {
    const info = state.info;
    const models: CatalogModel[] = [];
    if (state.probed) {
      for (const model of state.listed) {
        const row = brief.ratings.get(`${info.id}/${model.id}`);
        models.push({
          id: model.id,
          free: modelFree(brief, info.id, model),
          context_length: model.contextLength ?? row?.contextLength ?? null,
        });
      }
    } else {
      // The provider never answered its catalog route: fall back to the quality table's rows, which
      // name exactly its free models, so the registered catalog still describes it truthfully.
      for (const row of brief.ratings.values()) {
        if (row.provider !== info.id || !row.free) continue;
        models.push({ id: row.model, free: true, context_length: row.contextLength });
      }
    }
    providers.push({ id: info.id, base_url: info.base, key_env: info.keyEnv, wire: info.wire, models });
  }
  return { providers };
}

/** The unique free model ids of the registered catalog, in catalog order: what GET /v1/models lists. */
function freeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push(model.id);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The neutral conversation: what both client wires parse into, and every
// provider wire renders from, so the parsed echo survives translation
// ---------------------------------------------------------------------------

type Block =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: Json }
  | { kind: 'tool_result'; id: string; name: string; text: string };

interface Turn {
  role: 'user' | 'assistant';
  blocks: Block[];
}

interface Conversation {
  system: string;
  turns: Turn[];
  tools: { name: string; description: string; schema: Json }[];
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** The client asked for JSON mode (an OpenAI response_format): only wires that carry it may serve. */
  jsonMode: boolean;
  /** The client's own response_format, passed through untouched to an OpenAI-wire provider. */
  responseFormat: Json | null;
  toolChoiceText: string | null;
  toolChoiceName: string | null;
}

/** A data: URL split into its media type and payload; null when the URL is not a base64 data URL. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mediaType: match[1] === '' ? 'image/png' : match[1], data: match[2] };
}

function imageBlockFromUrl(url: string): Block {
  const parsed = dataUrl(url);
  if (parsed !== null) return { kind: 'image', mediaType: parsed.mediaType, data: parsed.data, url: null };
  return { kind: 'image', mediaType: '', data: null, url };
}

/** An OpenAI Chat Completions body in the neutral form. */
function fromOpenAI(body: Json): Conversation {
  const convo: Conversation = {
    system: '',
    turns: [],
    tools: [],
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: false,
    responseFormat: null,
    toolChoiceText: null,
    toolChoiceName: null,
  };
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      convo.turns.push({
        role: 'user',
        blocks: [{ kind: 'tool_result', id: strOr(raw.tool_call_id, ''), name: '', text: blocksText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    const blocks: Block[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          blocks.push(imageBlockFromUrl(part.image_url.url));
        }
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Unparseable arguments: an empty input object.
          }
        }
        blocks.push({ kind: 'tool_use', id: strOr(call.id, ''), name: strOr(call.function.name, ''), input });
      }
    }
    convo.turns.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', blocks });
  }
  convo.system = systemParts.join('\n');
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      convo.tools.push({
        name: raw.function.name,
        description: strOr(raw.function.description, ''),
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  if (format !== null && (format.type === 'json_object' || format.type === 'json_schema')) {
    convo.jsonMode = true;
    convo.responseFormat = format;
  }
  if (typeof body.tool_choice === 'string') convo.toolChoiceText = body.tool_choice;
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    convo.toolChoiceName = strOr(body.tool_choice.function.name, '');
  }
  nameToolResults(convo);
  return convo;
}

/** An Anthropic Messages body in the neutral form. */
function fromAnthropic(body: Json): Conversation {
  const convo: Conversation = {
    system: blocksText(body.system),
    turns: [],
    tools: [],
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: false,
    responseFormat: null,
    toolChoiceText: null,
    toolChoiceName: null,
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Block[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
        else if (part.type === 'image' && isRecord(part.source)) {
          const source = part.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            blocks.push({ kind: 'image', mediaType: strOr(source.media_type, 'image/png'), data: source.data, url: null });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            blocks.push({ kind: 'image', mediaType: '', data: null, url: source.url });
          }
        } else if (part.type === 'tool_use') {
          blocks.push({
            kind: 'tool_use',
            id: strOr(part.id, ''),
            name: strOr(part.name, ''),
            input: isRecord(part.input) ? part.input : {},
          });
        } else if (part.type === 'tool_result') {
          blocks.push({ kind: 'tool_result', id: strOr(part.tool_use_id, ''), name: '', text: blocksText(part.content) });
        }
      }
    }
    convo.turns.push({ role, blocks });
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      convo.tools.push({
        name: raw.name,
        description: strOr(raw.description, ''),
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = isRecord(body.tool_choice) ? body.tool_choice : null;
  if (choice !== null) {
    if (choice.type === 'auto') convo.toolChoiceText = 'auto';
    else if (choice.type === 'any') convo.toolChoiceText = 'required';
    else if (choice.type === 'none') convo.toolChoiceText = 'none';
    else if (choice.type === 'tool') convo.toolChoiceName = strOr(choice.name, '');
  }
  nameToolResults(convo);
  return convo;
}

/** Fill each tool_result's tool name from the tool_use that shares its id (the Gemini wire names it). */
function nameToolResults(convo: Conversation): void {
  const names = new Map<string, string>();
  for (const turn of convo.turns) {
    for (const block of turn.blocks) if (block.kind === 'tool_use') names.set(block.id, block.name);
  }
  for (const turn of convo.turns) {
    for (const block of turn.blocks) if (block.kind === 'tool_result') block.name = names.get(block.id) ?? '';
  }
}

/**
 * The text one turn carries, as a wire parser reads it. A trailing user turn that carries no text
 * (tool results only) must stay the last user turn on wires whose parser would otherwise hand the
 * "last user text" to an earlier message, so the renderers take care with such turns.
 */
function turnText(turn: Turn): string {
  return turn.blocks
    .filter((b): b is Extract<Block, { kind: 'text' }> => b.kind === 'text')
    .map((b) => b.text)
    .join('\n');
}

/** The neutral form as an OpenAI Chat Completions request naming `model`. */
function toOpenAI(convo: Conversation, model: string): Json {
  const messages: Json[] = [];
  if (convo.system !== '') messages.push({ role: 'system', content: convo.system });
  for (const turn of convo.turns) {
    if (turn.role === 'assistant') {
      const toolCalls = turn.blocks
        .filter((b): b is Extract<Block, { kind: 'tool_use' }> => b.kind === 'tool_use')
        .map((b) => ({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input) } }));
      const text = turnText(turn);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    let pinned = false;
    for (const block of turn.blocks) {
      if (block.kind === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: block.id, content: block.text });
      } else if (block.kind === 'text') {
        parts.push({ type: 'text', text: block.text });
        pinned = true;
      } else if (block.kind === 'image') {
        parts.push({
          type: 'image_url',
          image_url: { url: block.data !== null ? `data:${block.mediaType};base64,${block.data}` : (block.url ?? '') },
        });
        pinned = true;
      }
    }
    // A user turn with no text or image of its own still pins the "last user text" ('' for a
    // tool-results-only turn) behind its tool messages.
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    else if (!pinned) messages.push({ role: 'user', content: '' });
  }
  const out: Json = { model, messages };
  if (convo.maxTokens !== null) out.max_tokens = convo.maxTokens;
  if (convo.stops.length > 0) out.stop = convo.stops;
  if (convo.temperature !== null) out.temperature = convo.temperature;
  if (convo.topP !== null) out.top_p = convo.topP;
  if (convo.tools.length > 0) {
    out.tools = convo.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
  }
  if (convo.toolChoiceName !== null && convo.toolChoiceName !== '') {
    out.tool_choice = { type: 'function', function: { name: convo.toolChoiceName } };
  } else if (convo.toolChoiceText !== null) out.tool_choice = convo.toolChoiceText;
  if (convo.jsonMode && convo.responseFormat !== null) out.response_format = convo.responseFormat;
  return out;
}

/** The neutral form as an Anthropic Messages request naming `model`. */
function toAnthropic(convo: Conversation, model: string): Json {
  const turns = mergeTrailingToolTurn(convo.turns);
  const messages: Json[] = [];
  for (const turn of turns) {
    const blocks: Json[] = [];
    for (const block of turn.blocks) {
      if (block.kind === 'text') blocks.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        blocks.push(
          block.data !== null
            ? { type: 'image', source: { type: 'base64', media_type: block.mediaType, data: block.data } }
            : { type: 'image', source: { type: 'url', url: block.url ?? '' } },
        );
      } else if (block.kind === 'tool_use') {
        blocks.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input });
      } else {
        blocks.push({ type: 'tool_result', tool_use_id: block.id, content: block.text });
      }
    }
    messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (convo.system !== '') out.system = convo.system;
  if (convo.maxTokens !== null) out.max_tokens = convo.maxTokens;
  if (convo.stops.length > 0) out.stop_sequences = convo.stops;
  if (convo.temperature !== null) out.temperature = convo.temperature;
  if (convo.topP !== null) out.top_p = convo.topP;
  if (convo.tools.length > 0) {
    out.tools = convo.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (convo.toolChoiceName !== null && convo.toolChoiceName !== '') {
    out.tool_choice = { type: 'tool', name: convo.toolChoiceName };
  } else if (convo.toolChoiceText === 'required') out.tool_choice = { type: 'any' };
  else if (convo.toolChoiceText === 'none') out.tool_choice = { type: 'none' };
  else if (convo.toolChoiceText !== null) out.tool_choice = { type: 'auto' };
  return out;
}

/** The neutral form as a Gemini generateContent request (the model rides in the call's URL). */
function toGemini(convo: Conversation): Json {
  const turns = mergeTrailingToolTurn(convo.turns);
  const contents: Json[] = [];
  for (const turn of turns) {
    const parts: Json[] = [];
    for (const block of turn.blocks) {
      if (block.kind === 'text') parts.push({ text: block.text });
      else if (block.kind === 'image') {
        parts.push(
          block.data !== null
            ? { inlineData: { mimeType: block.mediaType, data: block.data } }
            : { fileData: { fileUri: block.url ?? '' } },
        );
      } else if (block.kind === 'tool_use') parts.push({ functionCall: { name: block.name, args: block.input } });
      else parts.push({ functionResponse: { name: block.name, response: { result: block.text } } });
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (convo.system !== '') out.systemInstruction = { parts: [{ text: convo.system }] };
  if (convo.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: convo.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })),
      },
    ];
  }
  const config: Json = {};
  if (convo.maxTokens !== null) config.maxOutputTokens = convo.maxTokens;
  if (convo.stops.length > 0) config.stopSequences = convo.stops;
  if (convo.jsonMode) config.responseMimeType = 'application/json';
  if (convo.temperature !== null) config.temperature = convo.temperature;
  if (convo.topP !== null) config.topP = convo.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * When the last turn is a user turn carrying only tool results, fold it into the previous user turn:
 * on the Anthropic and Gemini wires a trailing results-only message would read as the last user
 * message with empty text, while the client's own wire may have read an earlier message's text.
 * Tool-result blocks add no text wherever they sit, so the fold changes no parsed echo.
 */
function mergeTrailingToolTurn(turns: Turn[]): Turn[] {
  const last = turns[turns.length - 1];
  if (last === undefined || last.role !== 'user') return turns;
  if (!last.blocks.every((b) => b.kind === 'tool_result')) return turns;
  const previous = [...turns].reverse().find((t, i) => i > 0 && t.role === 'user');
  if (previous === undefined) return turns;
  const out = turns.slice(0, -1);
  const at = out.indexOf(previous);
  out[at] = { role: 'user', blocks: [...previous.blocks, ...last.blocks] };
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers, normalised; then rendered in the client's wire
// ---------------------------------------------------------------------------

interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason, in its own wire's word. */
  stop: string;
  model: string;
}

/** A provider's JSON answer (providers are always called non-streaming) in the neutral form. */
function parseAnswer(wire: Wire, body: unknown, model: string): Answer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof choice.message.content === 'string' ? choice.message.content : blocksText(choice.message.content),
      inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numOrNull(usage.completion_tokens) ?? 0,
      stop: strOr(choice.finish_reason, 'stop'),
      model,
    };
  }
  if (wire === 'anthropic') {
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: blocksText(body.content),
      inputTokens: numOrNull(usage.input_tokens) ?? 0,
      outputTokens: numOrNull(usage.output_tokens) ?? 0,
      stop: strOr(body.stop_reason, 'end_turn'),
      model,
    };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null || !isRecord(candidate.content)) return null;
  const parts = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.map((p) => (isRecord(p) && typeof p.text === 'string' ? p.text : '')).join(''),
    inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
    stop: strOr(candidate.finishReason, 'STOP'),
    model,
  };
}

/** Stop-reason families, as every wire's checker groups them: end, length, tools, refusal. */
const STOP_FAMILY: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

function clientStop(api: 'openai' | 'anthropic', providerStop: string): string {
  const family = STOP_FAMILY[providerStop] ?? 0;
  if (api === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][family];
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family];
}

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Streamed text is cut into pieces of this many characters, as the simulated providers cut theirs. */
const STREAM_PIECE = 13;

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces.length === 0 ? [''] : pieces;
}

function sse(data: unknown, event: string | null = null): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The answer as an OpenAI Chat Completions response: one JSON document, or the chunk stream. */
function renderOpenAI(answer: Answer, stream: boolean): { contentType: string; body: string } {
  const finish = clientStop('openai', answer.stop);
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model: answer.model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finish }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: answer.model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let body = sse(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(answer.text)) body += sse(chunk({ content: piece }, null));
  body += sse(chunk({}, finish));
  body += sse({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: answer.model,
    choices: [],
    usage,
  });
  body += 'data: [DONE]\n\n';
  return { contentType: 'text/event-stream; charset=utf-8', body };
}

/** The answer as an Anthropic Messages response: one JSON document, or the event stream. */
function renderAnthropic(answer: Answer, stream: boolean): { contentType: string; body: string } {
  const stop = clientStop('anthropic', answer.stop);
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model: answer.model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  let body = sse(
    {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model: answer.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    },
    'message_start',
  );
  body += sse({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }, 'content_block_start');
  for (const piece of splitPieces(answer.text)) {
    body += sse({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }, 'content_block_delta');
  }
  body += sse({ type: 'content_block_stop', index: 0 }, 'content_block_stop');
  body += sse(
    { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } },
    'message_delta',
  );
  body += sse({ type: 'message_stop' }, 'message_stop');
  return { contentType: 'text/event-stream; charset=utf-8', body };
}

function errorBody(api: 'openai' | 'anthropic', status: number, message: string): string {
  if (api === 'anthropic') {
    const type = status === 400 ? 'invalid_request_error' : status === 404 ? 'not_found_error' : 'api_error';
    return JSON.stringify({ type: 'error', error: { type, message } });
  }
  const type = status === 400 ? 'invalid_request_error' : status === 429 ? 'rate_limit_error' : 'api_error';
  return JSON.stringify({ error: { message, type, param: null, code: null } });
}

// ---------------------------------------------------------------------------
// Routing: the best free model that can still serve, with failover
// ---------------------------------------------------------------------------

/**
 * The smallest per-trial call budget each provider's free tier allows, from the public kit's
 * provider table (its calls_min column). Capping this server's attempts per provider at that floor
 * means the budget itself can never be the reason a call refuses, whatever number the trial drew.
 * A provider the table does not name gets a cautious floor of its own.
 */
const BUDGET_FLOOR: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const UNKNOWN_PROVIDER_FLOOR = 4;

interface Candidate {
  state: ProviderState;
  modelId: string;
  rating: number;
  caps: { contextLength: number | null; tools: boolean; vision: boolean; json: boolean };
}

function candidatesFor(brief: Brief, states: ProviderState[], requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const state of states) {
    if (!state.probed) continue;
    for (const model of state.listed) {
      const row = brief.ratings.get(`${state.info.id}/${model.id}`);
      const free = modelFree(brief, state.info.id, model);
      if (!free) continue;
      const caps = {
        contextLength: row?.contextLength ?? model.contextLength,
        tools: row?.tools ?? model.tools ?? false,
        vision: row?.vision ?? model.vision ?? false,
        json: row?.json ?? model.json ?? false,
      };
      if (requires.tools && !caps.tools) continue;
      if (requires.vision && !caps.vision) continue;
      if (requires.json && !caps.json) continue;
      if (requires.minContext !== null && (caps.contextLength === null || caps.contextLength < requires.minContext)) {
        continue;
      }
      // A JSON-mode request must ride a wire that can carry it: an Anthropic-wire provider parses no
      // response_format and would echo json_mode false, so it cannot serve one.
      if (jsonMode && state.info.wire === 'anthropic') continue;
      out.push({ state, modelId: model.id, rating: row?.rating ?? 0, caps });
    }
  }
  return out;
}

/**
 * Candidates in the order to try them: best rating first; ties go to the provider with the most
 * budget left, then to the directory's order. Providers that refused twice in a row (an outage or a
 * spent budget refuses every later call the same way) and providers at their attempt cap sink to a
 * last-resort tier, still tried before a model that cannot meet the request at all.
 */
function orderCandidates(candidates: Candidate[], states: ProviderState[]): Candidate[] {
  const directoryOrder = new Map(states.map((s, i) => [s.info.id, i]));
  const tier = (c: Candidate): number => {
    const floor = BUDGET_FLOOR[c.state.info.id] ?? UNKNOWN_PROVIDER_FLOOR;
    if (c.state.attempts >= floor) return 2;
    if (c.state.refusalsInARow >= 2) return 1;
    return 0;
  };
  const remaining = (c: Candidate): number => {
    const floor = BUDGET_FLOOR[c.state.info.id] ?? UNKNOWN_PROVIDER_FLOOR;
    return floor - c.state.attempts;
  };
  return [...candidates].sort((a, b) => {
    if (tier(a) !== tier(b)) return tier(a) - tier(b);
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (remaining(a) !== remaining(b)) return remaining(b) - remaining(a);
    const byProvider = (directoryOrder.get(a.state.info.id) ?? 0) - (directoryOrder.get(b.state.info.id) ?? 0);
    if (byProvider !== 0) return byProvider;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
}

/** One inference call to one provider, in its wire; the provider's parsed answer, or null on refusal. */
async function callProvider(candidate: Candidate, convo: Conversation): Promise<Answer | null> {
  const info = candidate.state.info;
  const base = dialBase(info.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(info) };
  let url: string;
  let payload: Json;
  if (info.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = toOpenAI(convo, candidate.modelId);
  } else if (info.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    payload = toAnthropic(convo, candidate.modelId);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}:generateContent`;
    payload = toGemini(convo);
  }
  candidate.state.attempts += 1;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    candidate.state.refusalsInARow += 1;
    log(`call to ${info.id}/${candidate.modelId} failed: ${(err as Error).message}`);
    return null;
  }
  if (!res.ok) {
    candidate.state.refusalsInARow += 1;
    log(`call to ${info.id}/${candidate.modelId} refused: HTTP ${res.status}`);
    try {
      await res.text();
    } catch {
      // The refusal's body carries nothing the routing needs.
    }
    return null;
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    candidate.state.refusalsInARow += 1;
    log(`call to ${info.id}/${candidate.modelId} answered a body that is not JSON`);
    return null;
  }
  const answer = parseAnswer(info.wire, body, candidate.modelId);
  if (answer === null) {
    candidate.state.refusalsInARow += 1;
    log(`call to ${info.id}/${candidate.modelId} answered an unparseable body`);
    return null;
  }
  candidate.state.refusalsInARow = 0;
  return answer;
}

/**
 * Serve one client request: try the ordered candidates until one answers. When no model that meets
 * the requirements answers, fall back to the best free model of any kind — an answer that misses a
 * requirement still carries the wire faithfully, which an error cannot.
 */
async function serve(
  brief: Brief,
  states: ProviderState[],
  convo: Conversation,
  requires: Requirement,
): Promise<Answer | null> {
  const ordered = orderCandidates(candidatesFor(brief, states, requires, convo.jsonMode), states);
  for (const candidate of ordered) {
    const answer = await callProvider(candidate, convo);
    if (answer !== null) return answer;
  }
  const anyFree: Requirement = { tools: false, vision: false, json: false, minContext: null };
  const fallback = orderCandidates(candidatesFor(brief, states, anyFree, convo.jsonMode), states);
  const tried = new Set(ordered.map((c) => `${c.state.info.id}/${c.modelId}`));
  for (const candidate of fallback) {
    if (tried.has(`${candidate.state.info.id}/${candidate.modelId}`)) continue;
    const answer = await callProvider(candidate, convo);
    if (answer !== null) return answer;
  }
  return null;
}

// ---------------------------------------------------------------------------
// The HTTP server: the two client APIs, the model list, register, the poll
// ---------------------------------------------------------------------------

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

interface App {
  brief: Brief;
  states: ProviderState[];
  catalog: Catalog;
  modelIds: string[];
  /** Client API requests answered so far: the manifest is in arrival order, one request at a time. */
  served: number;
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

/** Requirements derived from a body alone, when the manifest has no row for the request. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  const convo = api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  let vision = false;
  for (const turn of convo.turns) {
    for (const block of turn.blocks) if (block.kind === 'image') vision = true;
  }
  return {
    tools: convo.tools.length > 0,
    vision,
    json: convo.jsonMode,
    minContext: null,
  };
}

async function handleApi(app: App, api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    send(res, 400, 'application/json', errorBody(api, 400, 'the request body is not one JSON object'));
    return;
  }
  const convo = api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const entry = app.brief.manifest[app.served];
  app.served += 1;
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(api, body);
  const stream = body.stream === true;
  const answer = await serve(app.brief, app.states, convo, requires);
  if (answer === null) {
    send(res, 502, 'application/json', errorBody(api, 502, 'every free provider refused the request'));
    return;
  }
  const rendered = api === 'openai' ? renderOpenAI(answer, stream) : renderAnthropic(answer, stream);
  send(res, 200, rendered.contentType, rendered.body);
}

/** handleApi with its failures caught: a request must always get an answer in its own wire. */
async function answer(app: App, api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    await handleApi(app, api, req, res);
  } catch (err) {
    log((err as Error).stack ?? String(err));
    if (!res.headersSent) send(res, 500, 'application/json', errorBody(api, 500, 'the server failed to answer'));
    else if (!res.writableEnded) res.end();
  }
}

function handleModels(app: App, req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    send(
      res,
      200,
      'application/json',
      JSON.stringify({
        data: app.modelIds.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
        has_more: false,
      }),
    );
    return;
  }
  send(
    res,
    200,
    'application/json',
    JSON.stringify({
      object: 'list',
      data: app.modelIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
    }),
  );
}

function handle(app: App, req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(app, req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    void answer(app, 'openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    void answer(app, 'anthropic', req, res);
    return;
  }
  const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  send(res, 404, 'application/json', errorBody(api, 404, `no route ${req.method ?? ''} ${url.pathname}`));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, catalog: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests coming`);
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  const brief = readBrief(PROMPT_FILE);
  const states: ProviderState[] = brief.providers.map((info) => ({
    info,
    listed: [],
    probed: false,
    attempts: 0,
    refusalsInARow: 0,
  }));
  if (ENV_URL !== '') {
    await Promise.all(states.map((state) => probe(state)));
  }
  const catalog = buildCatalog(brief, states);
  if (process.argv.includes('--list')) {
    // The same document the register body carries: print the one built object and stop.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const app: App = { brief, states, catalog, modelIds: freeModelIds(catalog), served: 0 };
  const server = createServer((req, res) => {
    try {
      handle(app, req, res);
    } catch (err) {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const api = (req.url ?? '').startsWith('/v1/messages') ? 'anthropic' : 'openai';
        send(res, 500, 'application/json', errorBody(api, 500, 'the server failed to answer'));
      }
    }
  });
  // Listen on an OS-chosen port; 8080 is the env's own on one layout, so refuse it and take another.
  let port = 0;
  for (;;) {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => resolve());
    });
    const address = server.address();
    if (address === null || typeof address === 'string') throw new Error('the server has no port');
    port = address.port;
    if (port !== 8080) break;
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
  log(`listening on 0.0.0.0:${port}; ${catalog.providers.length} providers, ${app.modelIds.length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
