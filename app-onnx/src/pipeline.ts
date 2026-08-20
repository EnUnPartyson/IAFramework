/**
 * Pipeline de inferencia ONNX on-device: replica inference/pipeline_pytorch.py en el
 * navegador. Dos etapas: detector (perro/gato/ninguno) y, segun el resultado, el modelo
 * de raza correspondiente. Todo corre local con onnxruntime-web (wasm) — sin red.
 *
 * Hay DOS stacks completos: los pesos entrenados por PyTorch ("v1") y los entrenados por
 * TensorFlow ("v1_tf", portados a ONNX via SimpleCNN — ver portar_pesos_tf_a_onnx.py).
 * Ambos usan el mismo formato (NCHW + normalizacion simple), asi que este codigo no
 * distingue frameworks: solo cambia que pesos carga. Eso permite comparar TF vs PyTorch
 * en vivo, como en la app original.
 */
import * as ort from 'onnxruntime-web';

// los .wasm del runtime van empaquetados en public/ort/ (ver scripts/copy-ort.mjs):
// la app funciona offline y dentro de Capacitor, donde no hay CDN
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL + 'ort/';

export type Framework = 'pytorch' | 'tensorflow';

interface ModeloManifest {
  archivo: string;
  clases: string[];
  img_size: number;
  norm_mean: number[];
  norm_std: number[];
  umbral_no_identificada: number | null;
  temperatura: number;
}

type StackManifest = Record<'detector' | 'dog_breed' | 'cat_breed', ModeloManifest>;

interface Manifest {
  v1: StackManifest;
  v1_tf: StackManifest;
}

export interface Prediccion {
  framework: Framework;
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

interface Stack {
  detector: ModeloCargado;
  razas: Partial<Record<'perro' | 'gato', ModeloCargado>>;
}

// canvases de trabajo para el reescalado progresivo (ping-pong)
let tmpA: HTMLCanvasElement | null = null;
let tmpB: HTMLCanvasElement | null = null;

/**
 * Dibuja `fuente` en `destino` (s x s) con calidad comparable al resize de PIL.
 *
 * drawImage directo de 1920x1080 a 128x128 submuestrea SIN antialias y pierde una
 * enorme cantidad de detalle: el modelo recibe una imagen distinta a la que vio en
 * entrenamiento (donde el resize lo hacia PIL con filtrado) y la accuracy se derrumba.
 * El reescalado progresivo (reducir a la mitad por pasos) promedia pixeles en cada
 * paso y aproxima un resize con antialias real.
 *
 * `recortar`: toma el cuadrado central de la fuente en vez de estirarla. Para video
 * 16:9 evita aplastar al animal (el dataset se preparo desde fotos ~4:3, no 16:9).
 */
function dibujarEscalado(
  fuente: HTMLVideoElement | HTMLImageElement,
  destino: HTMLCanvasElement,
  s: number,
  recortar: boolean,
): void {
  const esVideo = fuente instanceof HTMLVideoElement;
  const fw = esVideo ? fuente.videoWidth : fuente.naturalWidth;
  const fh = esVideo ? fuente.videoHeight : fuente.naturalHeight;
  let sx = 0, sy = 0, sw = fw, sh = fh;
  if (recortar) {
    const lado = Math.min(fw, fh);
    sx = (fw - lado) / 2;
    sy = (fh - lado) / 2;
    sw = lado;
    sh = lado;
  }

  if (!tmpA) tmpA = document.createElement('canvas');
  if (!tmpB) tmpB = document.createElement('canvas');
  // primer paso: la fuente entra al canvas de trabajo reducida a la mitad como maximo
  let w = Math.max(s, Math.floor(sw / 2));
  let h = Math.max(s, Math.floor(sh / 2));
  tmpA.width = w;
  tmpA.height = h;
  const ctxA = tmpA.getContext('2d')!;
  ctxA.imageSmoothingEnabled = true;
  ctxA.imageSmoothingQuality = 'high';
  ctxA.drawImage(fuente, sx, sy, sw, sh, 0, 0, w, h);

  // reducir a la mitad hasta acercarse al tamano final (cada paso promedia pixeles)
  let actual = tmpA;
  let otro = tmpB;
  while (w > s * 2 || h > s * 2) {
    const nw = Math.max(s, Math.floor(w / 2));
    const nh = Math.max(s, Math.floor(h / 2));
    otro.width = nw;
    otro.height = nh;
    const ctx = otro.getContext('2d')!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(actual, 0, 0, w, h, 0, 0, nw, nh);
    const t = actual; actual = otro; otro = t;
    w = nw;
    h = nh;
  }

  destino.width = s;
  destino.height = s;
  const ctx = destino.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(actual, 0, 0, w, h, 0, 0, s, s);
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

/**
 * RGBA del canvas (img_size x img_size) -> [tensor NCHW normalizado, su espejo horizontal].
 * El espejo sale del mismo recorrido de pixeles (columna invertida): es el TTA que el
 * pipeline de la API aplica por defecto, replicado aca.
 */
export function preprocesar(
  pixeles: Uint8ClampedArray, info: ModeloManifest,
): [ort.Tensor, ort.Tensor] {
  const s = info.img_size;
  const plano = s * s;
  const datos = new Float32Array(3 * plano);
  const espejo = new Float32Array(3 * plano);
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const j = y * s + (s - 1 - x);
      const r = (pixeles[i * 4] / 255 - info.norm_mean[0]) / info.norm_std[0];
      const g = (pixeles[i * 4 + 1] / 255 - info.norm_mean[1]) / info.norm_std[1];
      const b = (pixeles[i * 4 + 2] / 255 - info.norm_mean[2]) / info.norm_std[2];
      datos[i] = r;
      datos[plano + i] = g;
      datos[2 * plano + i] = b;
      espejo[j] = r;
      espejo[plano + j] = g;
      espejo[2 * plano + j] = b;
    }
  }
  return [
    new ort.Tensor('float32', datos, [1, 3, s, s]),
    new ort.Tensor('float32', espejo, [1, 3, s, s]),
  ];
}

export class PipelineOnnx {
  private stacks: Partial<Record<Framework, Stack>> = {};

  /** Carga manifest + los 6 modelos (2 stacks x 3). Reporta progreso para la UI. */
  async cargar(onProgreso: (msg: string) => void): Promise<void> {
    const base = import.meta.env.BASE_URL + 'models/';
    onProgreso('Leyendo manifest...');
    const manifest: Manifest = await (await fetch(base + 'manifest.json')).json();

    const cargarUno = async (info: ModeloManifest, nombre: string): Promise<ModeloCargado> => {
      onProgreso(`Cargando ${nombre}...`);
      const sesion = await ort.InferenceSession.create(base + info.archivo, {
        executionProviders: ['wasm'],
      });
      return { sesion, info };
    };

    const cargarStack = async (sm: StackManifest, etiqueta: string): Promise<Stack> => ({
      detector: await cargarUno(sm.detector, `detector ${etiqueta}`),
      razas: {
        perro: await cargarUno(sm.dog_breed, `razas de perro ${etiqueta}`),
        gato: await cargarUno(sm.cat_breed, `razas de gato ${etiqueta}`),
      },
    });

    this.stacks.pytorch = await cargarStack(manifest.v1, '(PyTorch)');
    this.stacks.tensorflow = await cargarStack(manifest.v1_tf, '(TensorFlow)');
    onProgreso('Modelos listos');
  }

  private async probs(modelo: ModeloCargado, canvas: HTMLCanvasElement): Promise<Float32Array> {
    const s = modelo.info.img_size;
    // OJO: no tocar canvas.width/height aca. Asignarlos BORRA el canvas (por especificacion,
    // incluso con el mismo valor) y el frame que el llamador acaba de dibujar se pierde:
    // todas las inferencias saldrian de una imagen negra identica y la prediccion se congela.
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    const pixeles = ctx.getImageData(0, 0, s, s).data;
    // TTA con espejo horizontal, igual que PetPipeline en la API: promediar las dos
    // vistas reduce la varianza. Cuesta una pasada extra por modelo.
    const [tensor, tensorEspejo] = preprocesar(pixeles, modelo.info);
    const a = await modelo.sesion.run({ input: tensor });
    const pa = softmax(a.logits.data as Float32Array, modelo.info.temperatura);
    const b = await modelo.sesion.run({ input: tensorEspejo });
    const pb = softmax(b.logits.data as Float32Array, modelo.info.temperatura);
    const prom = new Float32Array(pa.length);
    for (let i = 0; i < pa.length; i++) prom[i] = (pa[i] + pb[i]) / 2;
    return prom;
  }

  /**
   * Clasifica un frame de video o una imagen quieta con el stack del framework indicado.
   * `recortar`: usar el cuadrado central (video 16:9, evita aplastar al animal);
   * false = estirar como el eval transform del entrenamiento (fotos de galeria).
   */
  async predecir(
    fuente: HTMLVideoElement | HTMLImageElement,
    canvas: HTMLCanvasElement,
    framework: Framework,
    recortar = false,
  ): Promise<Prediccion> {
    const stack = this.stacks[framework];
    if (!stack) throw new Error(`stack ${framework} no cargado`);
    const t0 = performance.now();

    const dibujar = (s: number) => dibujarEscalado(fuente, canvas, s, recortar);

    dibujar(stack.detector.info.img_size);
    const pDet = await this.probs(stack.detector, canvas);
    let idx = 0;
    for (let i = 1; i < pDet.length; i++) if (pDet[i] > pDet[idx]) idx = i;
    const especie = stack.detector.info.clases[idx] as Prediccion['especie'];

    const resultado: Prediccion = {
      framework,
      especie,
      especieConfianza: pDet[idx],
      raza: null,
      razaConfianza: null,
      razaIdentificada: true,
      topRazas: [],
      latenciaMs: 0,
    };

    const modeloRaza = especie === 'perro' || especie === 'gato' ? stack.razas[especie] : undefined;
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
