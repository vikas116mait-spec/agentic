import type { NextConfig } from "next";

const defaultAllowedDevOrigins = ["localhost", "127.0.0.1", "10.10.110.22", "10.120.89.155"];
const configuredAllowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  typedRoutes: true,
  output: "standalone",
  allowedDevOrigins: Array.from(new Set([...defaultAllowedDevOrigins, ...configuredAllowedDevOrigins]))
  // NOTE: `NEXT_PUBLIC_PYTHON_API_URL` is no longer baked into the client
  // bundle. The browser goes through the same-origin proxy at
  // `/api/py/[...path]` (see `app/api/py/[...path]/route.ts`), which resolves
  // the real Python API from the server-only `PYTHON_API_URL` env var at
  // request time. This avoids the "baked stale URL" class of bugs.
};

export default nextConfig;
