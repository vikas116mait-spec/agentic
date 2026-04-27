/**
 * Same-origin proxy to the Python FastAPI service.
 *
 * Why this exists:
 *   Every `pythonApiFetch` call used to go *directly* from the browser to
 *   the Python API host (e.g. http://127.0.0.1:12400). That works only when
 *   the browser sits on the same machine as the Python API. It breaks in
 *   every other deployment shape:
 *     - LAN browsers get "connection refused" for the dev machine's loopback.
 *     - HTTPS Vercel pages get "NetworkError when attempting to fetch
 *       resource" because the browser blocks mixed (http) requests.
 *     - Mobile / hosted previews can't route to a private IP at all.
 *
 * Solution:
 *   All client code now targets `/api/py/*` on the Next.js origin, and this
 *   route forwards every request (method, headers, body, query string) to
 *   `PYTHON_API_URL` on the server side. Response body, status, and headers
 *   are streamed back unchanged. Since the browser only ever talks to the
 *   Next.js origin, CORS + mixed-content issues disappear entirely.
 *
 *   `PYTHON_API_URL` is a server-only env var -- it is never bundled into
 *   the client, so rotating the backend URL doesn't require redeploying
 *   the frontend.
 */

import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Allow long-running downloads / streaming responses.
export const maxDuration = 300;

function pythonApiBaseUrl(): string {
  const raw =
    process.env.PYTHON_API_URL ??
    process.env.NEXT_PUBLIC_PYTHON_API_URL ??
    "http://127.0.0.1:8001";
  return raw.replace(/\/+$/, "");
}

// Headers we must not forward. "host" would point at the browser's origin,
// "cookie" leaks Next.js auth to the Python API (it doesn't speak them), and
// the rest are hop-by-hop headers that fetch manages itself.
const DROP_REQUEST_HEADERS = new Set([
  "host",
  "cookie",
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "content-length",
]);

function buildForwardHeaders(src: Headers): Headers {
  const out = new Headers();
  src.forEach((value, key) => {
    if (!DROP_REQUEST_HEADERS.has(key.toLowerCase())) {
      out.set(key, value);
    }
  });
  return out;
}

async function proxy(
  req: NextRequest,
  ctx: { params: Promise<{ path?: string[] }> },
): Promise<Response> {
  const { path = [] } = await ctx.params;
  const search = req.nextUrl.search;
  const target = `${pythonApiBaseUrl()}/${path.join("/")}${search}`;

  const method = req.method.toUpperCase();
  const bodyless = method === "GET" || method === "HEAD";

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers: buildForwardHeaders(req.headers),
    redirect: "manual",
  };

  if (!bodyless) {
    // Stream the body through so large multipart uploads (datasets) don't
    // get buffered into Node memory.
    init.body = req.body;
    init.duplex = "half";
  }

  let response: Response;
  try {
    response = await fetch(target, init);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(
      JSON.stringify({
        error: {
          code: "PYTHON_API_UNREACHABLE",
          message:
            `Could not reach the Python API at ${pythonApiBaseUrl()} from the ` +
            `Next.js server. Verify PYTHON_API_URL and make sure the service ` +
            `is running. Details: ${message}`,
        },
      }),
      {
        status: 502,
        headers: { "content-type": "application/json" },
      },
    );
  }

  const outHeaders = new Headers(response.headers);
  // Let Next/undici manage framing of the forwarded body.
  outHeaders.delete("content-length");
  outHeaders.delete("content-encoding");
  outHeaders.delete("transfer-encoding");
  outHeaders.delete("connection");

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: outHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;
