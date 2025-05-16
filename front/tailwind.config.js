/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Source Code Pro', 'monospace'],
      },
      colors: {
        // Custom colors can be added here if needed
      },
      animation: {
        'bounce-slow': 'bounce 3s linear infinite',
      },
    },
  },
  plugins: [],
}