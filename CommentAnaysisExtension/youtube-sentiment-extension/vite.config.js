import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      input: {
        popup: 'popup.html',
        demo: 'demo.html',
        popupjs:"popup.js"
      }
    },
    outDir: 'dist'
  }
})
