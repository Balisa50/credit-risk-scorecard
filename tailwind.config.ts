import type { Config } from "tailwindcss";

export default {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0a0a0a",
        surface: "#141414",
        border: "#262626",
        cyan: "#00F0FF",
        risk: {
          low: "#22c55e",
          medium: "#eab308",
          high: "#ef4444",
          extreme: "#dc2626",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
