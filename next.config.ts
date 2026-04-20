import type { NextConfig } from "next";

const defaultAllowedDevOrigins = ["localhost", "127.0.0.1", "10.10.110.22", "10.120.89.155"];
const configuredAllowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  typedRoutes: true,
  output: "standalone",
  allowedDevOrigins: Array.from(new Set([...defaultAllowedDevOrigins, ...configuredAllowedDevOrigins])),
  env: {
    NEXT_PUBLIC_PYTHON_API_URL: process.env.NEXT_PUBLIC_PYTHON_API_URL ?? process.env.PYTHON_API_URL ?? "http://127.0.0.1:8001"
  }
};

export default nextConfig;
