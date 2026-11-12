import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const __dirname = dirname(fileURLToPath(import.meta.url));

const config = [
  ...nextVitals,
  ...nextTs,
  {
    settings: {
      next: {
        rootDir: __dirname,
      },
    },
    rules: {
      "@next/next/no-img-element": "off",
    },
  },
];

export default config;
