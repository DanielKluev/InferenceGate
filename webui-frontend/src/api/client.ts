/**
 * API client for InferenceGate WebUI
 */

const API_BASE = '/api';

export interface CacheEntry {
  id: string;
  model: string | null;
  path: string;
  method: string;
  status_code: number;
  is_streaming: boolean;
  temperature: number | null;
  prompt_hash: string | null;
}

export interface CacheEntryDetail {
  id: string;
  model: string | null;
  temperature: number | null;
  prompt_hash: string | null;
  request: {
    method: string;
    path: string;
    headers: Record<string, string>;
    body: any;
    query_params: Record<string, string> | null;
  };
  response: {
    status_code: number;
    headers: Record<string, string>;
    body: any;
    chunks: string[] | null;
    is_streaming: boolean;
  };
}

export interface CacheStats {
  total_entries: number;
  total_size_bytes: number;
  streaming_responses: number;
  entries_by_model: Record<string, number>;
}

export interface Config {
  mode: string;
  upstream_url: string | null;
  host: string;
  port: number;
  cache_dir: string;
}

export async function getCacheList(): Promise<CacheEntry[]> {
  const response = await fetch(`${API_BASE}/cache`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cache list: ${response.statusText}`);
  }
  return response.json();
}

export async function getCacheEntry(id: string): Promise<CacheEntryDetail> {
  const response = await fetch(`${API_BASE}/cache/${id}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cache entry: ${response.statusText}`);
  }
  return response.json();
}

export async function getCacheStats(): Promise<CacheStats> {
  const response = await fetch(`${API_BASE}/stats`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cache stats: ${response.statusText}`);
  }
  return response.json();
}

export async function getConfig(): Promise<Config> {
  const response = await fetch(`${API_BASE}/config`);
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.statusText}`);
  }
  return response.json();
}
