import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  // host: true para poder abrir el dev server desde el celular en la misma red
  server: { host: true, port: 5174 },
});
