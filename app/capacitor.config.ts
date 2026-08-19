import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.iaframework.mascotas',
  appName: 'Clasificador de Mascotas',
  webDir: 'dist',
  android: {
    // el WebView de Capacitor sirve la app desde https://localhost, asi que pedir a una
    // API http:// cuenta como contenido mixto y se bloquea sin esto
    allowMixedContent: true,
  },
  server: {
    // Android bloquea el trafico HTTP sin cifrar desde Android 9. La API corre en
    // http://<ip-de-la-laptop>:8000 dentro de la red local, asi que hay que permitirlo.
    // Para produccion lo correcto seria poner la API detras de HTTPS.
    cleartext: true,
  },
};

export default config;
