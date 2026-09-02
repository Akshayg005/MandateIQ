import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 4318, not Vite's default 5173, and next to the dashboard's 4317 so the
// two dev servers can run side by side.
//
// Windows reserves dynamic TCP ranges that move on reboot, and this project
// has already lost time to one: dashboard/vite.config.ts and .env both record
// a day when 5432 and 5173 were inside reserved blocks and neither Postgres
// nor Vite would bind. Those particular ranges are no longer reserved on this
// machine -- `netsh interface ipv4 show excludedportrange protocol=tcp` --
// which is the point: a default that works today is not a default that works
// after the next reboot.
//
// strictPort so a reserved or busy port fails loudly here rather than
// silently landing on a different one than run.ps1 and the docs name.
const PORT = 4318

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { port: PORT, strictPort: true },
  preview: { port: PORT, strictPort: true },
})
