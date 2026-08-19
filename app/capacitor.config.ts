import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.iaframework.mascotas',
  appName: 'Clasificador de Mascotas',
  webDir: 'dist',
  server: {
    // Android bloquea el trafico HTTP sin cifrar desde Android 9. La API corre en
    // http://<ip-de-la-laptop>:8000 dentro de la red local, asi que hay que permitirlo.
    // Para produccion lo correcto seria poner la API detras de HTTPS.
    cleartext: true,
  },
};

export default config;
