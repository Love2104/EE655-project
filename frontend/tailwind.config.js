/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        pitch: {
          950: "#07111f",
          900: "#0d1729",
          800: "#132238",
          700: "#1b314a",
          600: "#23516d"
        },
        neon: {
          cyan: "#4df6ff",
          lime: "#b7ff7a",
          amber: "#ffb347",
          coral: "#ff785a"
        }
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(77,246,255,0.08), 0 20px 60px rgba(0,0,0,0.35)"
      },
      backgroundImage: {
        "pitch-grid": "radial-gradient(circle at 1px 1px, rgba(148,163,184,0.14) 1px, transparent 0)"
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["IBM Plex Sans", "sans-serif"]
      }
    }
  },
  plugins: []
};
