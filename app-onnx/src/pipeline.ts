/**
 * Pipeline de inferencia ONNX on-device: replica inference/pipeline_pytorch.py en el
 * navegador. Dos etapas: detector (perro/gato/ninguno) y, segun el resultado, el modelo
 * de raza correspondiente. Todo corre local con onnxruntime-web (wasm) — sin red.
 *
 * El preprocesamiento reproduce get_eval_transforms(): resize a img_size (lo hace el
 * canvas al dibujar el frame) + normalizacion con la media/desvio del manifest.
 */
import * as ort from 'onnxruntime-web';

// los .wasm del runtime van empaquetados en public/ort/ (ver scripts/copy-ort.mjs):
// la app funciona offline y dentro de Capacitor, donde no hay CDN
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL + 'ort/';

interface ModeloManifest {
  archivo: string;
  clases: string[];
  img_size: number;
  norm_mean: number[];
  norm_std: number[];
  umbral_no_identificada: number | null;
  temperatura: number;
}

interface Manifest {
  v1: Record<'detector' | 'dog_breed' | 'cat_breed', ModeloManifest>;
}

export interface Prediccion {
  especie: 'perro' | 'gato' | 'ninguno';
  especieConfianza: number;
  raza: string | null;
  razaConfianza: number | null;
  razaIdentificada: boolean;
  topRazas: [string, number][];
  /** milisegundos que tardo la inferencia completa (detector + raza) */
  latenciaMs: number;
}

interface ModeloCargado {
  sesion: ort.InferenceSession;
  info: ModeloManifest;
}

function softmax(logits: Float32Array, temperatura: number): Float32Array {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v / temperatura);
  const exps = new Float32Array(logits.length);
  let suma = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] / temperatura - max);
    suma += exps[i];
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= suma;
  return exps;
}

/** RGBA del canvas (img_size x img_size) -> tensor NCHW float32 normalizado. */
export function preprocesar(pixeles: Uint8ClampedArray, info: ModeloManifest): ort.Tensor {
  const s = info.img_size;
  const plano = s * s;
  const datos = new Float32Array(3 * plano);
  for (let i = 0; i < plano; i++) {
    const r = pixeles[i * 4] / 255;
    const g = pixeles[i * 4 + 1] / 255;
    const b = pixeles[i * 4 + 2] / 255;
    datos[i] = (r - info.norm_mean[0]) / info.norm_std[0];
    datos[plano + i] = (g - info.norm_mean[1]) / info.norm_std[1];
    datos[2 * plano + i] = (b - info.norm_mean[2]) / info.norm_std[2];
  }
  return new ort.Tensor('float32', datos, [1, 3, s, s]);
}

export class PipelineOnnx {
  private detector!: ModeloCargado;
  private razas: Partial<Record<'perro' | 'gato', ModeloCargado>> = {};

  /** Carga manifest + los 3 modelos. Reportar progreso permite mostrarlo en la UI. */
  async cargar(onProgreso: (msg: string) => void): Promise<void> {
    const base = import.meta.env.BASE_URL + 'models/';
    onProgreso('Leyendo manifest...');
    const manifest: Manifest = await (await fetch(base + 'manifest.json')).json();

    const cargarUno = async (info: ModeloManifest, nombre: string): Promise<ModeloCargado> => {
      onProgreso(`Cargando ${nombre} (${info.clases.length} clases)...`);
      const sesion = await ort.InferenceSession.create(base + info.archivo, {
        executionProviders: ['wasm'],
      });
      return { sesion, info };
    };

    this.detector = await cargarUno(manifest.v1.detector, 'detector');
    this.razas.perro = await cargarUno(manifest.v1.dog_breed, 'razas de perro');
    this.razas.gato = await cargarUno(manifest.v1.cat_breed, 'razas de gato');
    onProgreso('Modelos listos');
  }

  private async probs(modelo: ModeloCargado, canvas: HTMLCanvasElement): Promise<Float32Array> {
    const s = modelo.info.img_size;
    canvas.width = s;
    canvas.height = s;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    // el llamador ya dibujo el frame en este canvas al tamano correcto
    const pixeles = ctx.getImageData(0, 0, s, s).data;
    const tensor = preprocesar(pixeles, modelo.info);
    const salida = await modelo.sesion.run({ input: tensor });
    return softmax(salida.logits.data as Float32Array, modelo.info.temperatura);
  }

  /**
   * Clasifica el frame actual del video. `canvas` es un canvas de trabajo reutilizable;
   * esta funcion lo redimensiona y dibuja el frame por cada modelo que corre.
   */
  async predecir(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Prediccion> {
    const t0 = performance.now();

    const dibujar = (s: number) => {
      canvas.width = s;
      canvas.height = s;
      // Resize((s, s)) del eval transform: estira el frame completo, sin recortar
      canvas.getContext('2d', { willReadFrequently: true })!.drawImage(video, 0, 0, s, s);
    };

    dibujar(this.detector.info.img_size);
    const pDet = await this.probs(this.detector, canvas);
    let idx = 0;
    for (let i = 1; i < pDet.length; i++) if (pDet[i] > pDet[idx]) idx = i;
    const especie = this.detector.info.clases[idx] as Prediccion['especie'];

    const resultado: Prediccion = {
      especie,
      especieConfianza: pDet[idx],
      raza: null,
      razaConfianza: null,
      razaIdentificada: true,
      topRazas: [],
      latenciaMs: 0,
    };

    const modeloRaza = especie === 'perro' || especie === 'gato' ? this.razas[especie] : undefined;
    if (modeloRaza) {
      dibujar(modeloRaza.info.img_size);
      const pRaza = await this.probs(modeloRaza, canvas);
      const orden = [...pRaza.keys()].sort((a, b) => pRaza[b] - pRaza[a]).slice(0, 3);
      resultado.topRazas = orden.map((i) => [modeloRaza.info.clases[i], pRaza[i]]);
      resultado.raza = resultado.topRazas[0][0];
      resultado.razaConfianza = resultado.topRazas[0][1];
      const umbral = modeloRaza.info.umbral_no_identificada ?? 0.45;
      resultado.razaIdentificada = resultado.razaConfianza >= umbral;
    }

    resultado.latenciaMs = performance.now() - t0;
    return resultado;
  }
}
