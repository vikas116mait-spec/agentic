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
};

export default nextConfig;
