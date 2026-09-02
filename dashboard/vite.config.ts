import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 4317, not Vite's default 5173. On 2026-08-31 this machine's
// Windows/Hyper-V dynamic range reserved 5141-5240, so `vite --port 5173`
// died with EACCES on ::1 before serving anything; the same day 5432 fell
// inside the reserved 5341-5440 block, which is why this project's Postgres
// container publishes 15432.
//
// Those ranges are no longer reserved here -- check with `netsh interface
// ipv4 show excludedportrange protocol=tcp` -- because Windows reshuffles
// them on reboot. That is the reason to pin rather than the reason to stop:
// a default that binds today is not a default that binds after the next
// restart.
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
