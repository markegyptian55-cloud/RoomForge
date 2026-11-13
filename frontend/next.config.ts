import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  devIndicators: {
    position: "top-right",
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
