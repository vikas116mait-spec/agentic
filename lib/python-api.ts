export type PythonApiError = {
  error?: {
    code?: string;
    message?: string;
    details?: unknown[];
  };
};

/**
 * URL prefix the **browser** uses for every Python API call. It targets the
 * Next.js same-origin proxy at `app/api/py/[...path]/route.ts`, which forwards
 * to the real Python API on the server side. Keeping this same-origin fixes
 * three long-standing bugs:
 *   - CORS preflight failures when the browser and API are on different hosts.
 *   - Mixed-content blocks when an HTTPS Vercel page tried to hit an HTTP
 *     localhost backend.
 *   - "Connection refused" when a browser on another machine tried to reach
 *     the dev-box loopback address baked into `NEXT_PUBLIC_PYTHON_API_URL`.
 */
const CLIENT_PYTHON_API_PREFIX = "/api/py";

const SERVER_DEFAULT_PYTHON_API_URL = "http://127.0.0.1:8001";

/**
 * Fully-qualified base URL used when this module runs on the server (SSR
 * components, route handlers, server actions). `PYTHON_API_URL` is the
 * canonical server-only variable; `NEXT_PUBLIC_PYTHON_API_URL` is still
 * accepted for backwards compatibility but should be considered deprecated
 * now that nothing reads it from the client bundle.
 */
function resolveServerPythonApiUrl(): string {
  const raw =
    process.env.PYTHON_API_URL ??
    process.env.NEXT_PUBLIC_PYTHON_API_URL ??
    SERVER_DEFAULT_PYTHON_API_URL;
  return raw.replace(/\/+$/, "");
}

const IS_SERVER = typeof window === "undefined";

/**
 * Return a base URL that, when concatenated with a Python API path (e.g.
 * `/datasets/${id}`), yields a URL reachable from the current execution
 * context.
 *
 * - Server context (SSR, route handlers, `app/*` data fetching): returns the
 *   full `PYTHON_API_URL`, e.g. `http://127.0.0.1:12400`.
 * - Browser context: returns `/api/py`, so fetches stay same-origin and the
 *   Next.js proxy handles the hop to the Python API.
 *
 * Callers that build `<a href>` or `window.location.assign` URLs (downloads)
 * automatically work in both contexts because the browser treats
 * `/api/py/...` as a same-origin URL it can navigate to.
 */
export function getPythonApiBaseUrl(): string {
  return IS_SERVER ? resolveServerPythonApiUrl() : CLIENT_PYTHON_API_PREFIX;
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
