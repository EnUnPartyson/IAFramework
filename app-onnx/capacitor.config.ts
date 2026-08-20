import type { CapacitorConfig } from '@capacitor/cli';

// App "Live": inferencia ONNX on-device con camara en vivo. NO usa red: los modelos van
// empaquetados en la app, por eso no necesita cleartext ni permisos de internet especiales.
const config: CapacitorConfig = {
  appId: 'com.iaframework.mascotas.live',
  appName: 'Mascotas Live',
  webDir: 'dist',
};

export default config;
