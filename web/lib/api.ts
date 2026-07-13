import type {
  LogsResponse,
  Provider,
  ResultResponse,
  RunResponse,
  StatusResponse,
  UploadResponse,
} from "./types";

// Backend base URL. Set NEXT_PUBLIC_API_BASE_URL to the Cloud Run URL in production;
// defaults to the local FastAPI dev server.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  return asJson<UploadResponse>(res);
}

export async function runPipeline(body: {
  upload_id: string;
  business_problem: string;
  provider: Provider;
  api_key: string;
  model_name: string;
}): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return asJson<RunResponse>(res);
}

export async function getStatus(id: string): Promise<StatusResponse> {
  return asJson<StatusResponse>(await fetch(`${API_BASE}/pipeline/${id}/status`));
}

export async function getLogs(id: string, offset = 0): Promise<LogsResponse> {
  return asJson<LogsResponse>(
    await fetch(`${API_BASE}/pipeline/${id}/logs?offset=${offset}`),
  );
}

export async function getResult(id: string): Promise<ResultResponse> {
  return asJson<ResultResponse>(await fetch(`${API_BASE}/pipeline/${id}/result`));
}

export function artifactUrl(id: string, file: string): string {
  return `${API_BASE}/pipeline/${id}/artifacts/${file}`;
}
