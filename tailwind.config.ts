import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#f6f4ee",
        ink: "#111111",
        brand: {
          DEFAULT: "#1f6f5f",
          foreground: "#f7fbf9"
        },
        accent: "#e7b24b",
        danger: "#c74634",
        muted: "#ebe6db",
        panel: "#fffdf7"
      },
      fontFamily: {
        sans: ["'Space Grotesk'", "ui-sans-serif", "system-ui"],
        display: ["'Fraunces'", "Georgia", "serif"]
      },
      boxShadow: {
        panel: "0 14px 50px rgba(17, 17, 17, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
