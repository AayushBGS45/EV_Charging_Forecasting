/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "#0e1511",
        surface: "#0e1511",
        "surface-container-lowest": "#09100c",
        "surface-container-low": "#161d19",
        "surface-container": "#1a211d",
        "surface-container-high": "#242c27",
        "surface-container-highest": "#2f3632",
        primary: "#4edea3",
        "primary-container": "#10b981",
        "on-primary": "#003824",
        secondary: "#adc6ff",
        "secondary-container": "#0566d9",
        tertiary: "#ffb95f",
        "tertiary-container": "#e29100",
        "on-surface": "#dde4dd",
        "on-surface-variant": "#bbcabf",
        outline: "#86948a",
        "outline-variant": "#3c4a42",
      },
      fontFamily: {
        headline: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        label: ["'Public Sans'", "sans-serif"],
      },
    },
  },
  plugins: [],
}

