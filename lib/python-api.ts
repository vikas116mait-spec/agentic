export type PythonApiError = {
  error?: {
    code?: string;
    message?: string;
    details?: unknown[];
  };
};

const DEFAULT_PYTHON_API_URL = "http://127.0.0.1:8001";

export function getPythonApiBaseUrl() {
  return (
    process.env.NEXT_PUBLIC_PYTHON_API_URL ??
    process.env.PYTHON_API_URL ??
    DEFAULT_PYTHON_API_URL
  );
}

export async function pythonApiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getPythonApiBaseUrl()}${path}`, init);
  const contentType = response.headers.get("content-type") ?? "";
  const payload: (T & PythonApiError) | null = contentType.includes("application/json")
    ? ((await response.json()) as T & PythonApiError)
    : null;

  if (!response.ok) {
    throw new Error(payload?.error?.message ?? `Request failed with status ${response.status}`);
  }

  return payload as T;
}
