import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 4317, not Vite's default 5173. This machine's Windows/Hyper-V dynamic
// range reserves 5141-5240 (`netsh interface ipv4 show excludedportrange
// protocol=tcp`), so `vite --port 5173` dies with EACCES on ::1 before it
// serves anything. Same class of problem, and the same fix, as the Postgres
// port note in .env -- 5432 is inside the reserved 5341-5440 block, which is
// why this project's container publishes 15432.
//
// strictPort so a reserved or busy port fails loudly here rather than
// silently landing on a different one than the docs and run.ps1 name.
const PORT = 4317

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { port: PORT, strictPort: true },
  preview: { port: PORT, strictPort: true },
})
