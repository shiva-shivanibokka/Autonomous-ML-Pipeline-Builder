import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server output, so web/Dockerfile can ship a small runtime
  // image for `docker compose up`. Vercel ignores this and builds normally.
  output: "standalone",
};

export default nextConfig;
