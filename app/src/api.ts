/**
 * Cliente de la API de clasificacion (inference/server.py).
 *
 * La URL del servidor la configura el usuario en la app y se guarda con Preferences:
 * en el celular hay que apuntar a la IP de la laptop en la red WiFi, que cambia segun
 * la red, asi que no puede ir hardcodeada.
 */
import { Preferences } from '@capacitor/preferences';

const CLAVE_URL = 'apiBaseUrl';
const URL_POR_DEFECTO = 'http://192.168.1.100:8000';

export interface Prediccion {
  especie: 'perro' | 'gato' | 'ninguno';
  especie_confianza: number;
  raza: string | null;
  raza_confianza: number | null;
  raza_identificada: boolean;
  top_razas: [string, number][];
  resumen: string;
}

export interface Estado {
  ok: boolean;
  modo: string;
  error: string | null;
}

export async function obtenerUrlApi(): Promise<string> {
  const { value } = await Preferences.get({ key: CLAVE_URL });
  return value ?? URL_POR_DEFECTO;
}

export async function guardarUrlApi(url: string): Promise<void> {
  await Preferences.set({ key: CLAVE_URL, value: url.replace(/\/+$/, '') });
}

/** Falla rapido si el servidor no responde, en vez de dejar la UI colgada. */
async function fetchConTimeout(url: string, opciones: RequestInit = {}, ms = 20000): Promise<Response> {
  const control = new AbortController();
  const id = setTimeout(() => control.abort(), ms);
  try {
    return await fetch(url, { ...opciones, signal: control.signal });
  } finally {
    clearTimeout(id);
  }
}

export async function verificarEstado(): Promise<Estado> {
  const base = await obtenerUrlApi();
  const res = await fetchConTimeout(`${base}/health`, {}, 8000);
  if (!res.ok) throw new Error(`El servidor respondio ${res.status}`);
  return res.json();
}

export async function clasificar(fotoBase64: string): Promise<Prediccion> {
  const base = await obtenerUrlApi();

  // Capacitor devuelve la foto en base64; la API espera multipart/form-data
  const binario = atob(fotoBase64);
  const bytes = new Uint8Array(binario.length);
  for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);

  const formulario = new FormData();
  formulario.append('file', new Blob([bytes], { type: 'image/jpeg' }), 'foto.jpg');

  const res = await fetchConTimeout(`${base}/predict`, { method: 'POST', body: formulario }, 30000);
  if (!res.ok) {
    const detalle = await res.json().catch(() => ({ detail: `error ${res.status}` }));
    throw new Error(detalle.detail ?? `El servidor respondio ${res.status}`);
  }
  return res.json();
}
