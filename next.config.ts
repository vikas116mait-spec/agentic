import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typedRoutes: true,
  output: "standalone",
  allowedDevOrigins: ["10.120.89.155"]
};

export default nextConfig;
