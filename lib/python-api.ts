export type PythonApiError = {
  error?: {
    code?: string;
    message?: string;
    details?: unknown[];
  };
};

export function getPythonApiBaseUrl() {
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8001`;
  }

  return process.env.PYTHON_API_URL ?? "http://127.0.0.1:8001";
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
