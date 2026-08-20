// Copia los binarios wasm de onnxruntime-web a public/ort/ para que la app funcione
// 100% offline (sin CDN): el runtime los busca en la ruta que fija ort.env.wasm.wasmPaths.
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const raiz = dirname(dirname(fileURLToPath(import.meta.url)));
const origen = join(raiz, 'node_modules', 'onnxruntime-web', 'dist');
const destino = join(raiz, 'public', 'ort');
mkdirSync(destino, { recursive: true });
let n = 0;
for (const f of readdirSync(origen)) {
  if (f.endsWith('.wasm') || f.endsWith('.mjs')) {
    cpSync(join(origen, f), join(destino, f));
    n++;
  }
}
console.log(`copy-ort: ${n} archivos copiados a public/ort/`);
