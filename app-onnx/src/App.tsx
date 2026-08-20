/**
 * Mascotas Live: camara en vivo + inferencia ONNX on-device. Sin servidor, sin red:
 * los modelos V1 van empaquetados y clasifican el video continuamente, como hace
 * inference/predict_camera.py en la laptop con OpenCV.
 */
import {
  IonApp, IonBadge, IonButton, IonContent, IonHeader, IonIcon, IonLabel, IonSegment,
  IonSegmentButton, IonSpinner, IonTitle, IonToolbar, setupIonicReact,
} from '@ionic/react';
import { cameraReverse, images, pause, play, videocam } from 'ionicons/icons';
import { useEffect, useRef, useState } from 'react';
import { PipelineOnnx, type Framework, type Prediccion } from './pipeline';

import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import './App.css';

setupIonicReact();

const EMOJI: Record<string, string> = { perro: '\u{1F436}', gato: '\u{1F431}', ninguno: '\u{1F6AB}' };

type Estado = 'cargando' | 'listo' | 'sin-camara' | 'error';
type Modo = Framework | 'comparar';
const CHIP: Record<Framework, string> = { pytorch: 'PyTorch', tensorflow: 'TensorFlow' };

const App: React.FC = () => {
  const [estado, setEstado] = useState<Estado>('cargando');
  const [progreso, setProgreso] = useState('Iniciando...');
  const [predicciones, setPredicciones] = useState<Prediccion[]>([]);
  const [pausado, setPausado] = useState(false);
  const [camaraTrasera, setCamaraTrasera] = useState(true);
  const [modo, setModo] = useState<Modo>('pytorch');
  // dataURL de una foto elegida de la galeria; mientras exista, el bucle en vivo se frena
  // y la clasificacion corre sobre la imagen quieta
  const [foto, setFoto] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const inputFotoRef = useRef<HTMLInputElement>(null);
  const fotoRef = useRef<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pipelineRef = useRef<PipelineOnnx | null>(null);
  const pausadoRef = useRef(false);
  const modoRef = useRef<Modo>('pytorch');

  pausadoRef.current = pausado;
  modoRef.current = modo;
  fotoRef.current = foto;

  // ---- carga de modelos (una vez) ----
  useEffect(() => {
    const pipeline = new PipelineOnnx();
    pipelineRef.current = pipeline;
    canvasRef.current = document.createElement('canvas');
    pipeline
      .cargar(setProgreso)
      .then(() => setEstado('listo'))
      .catch((e) => {
        console.error(e);
        setProgreso(`No se pudieron cargar los modelos: ${(e as Error).message}`);
        setEstado('error');
      });
  }, []);

  // ---- camara ----
  useEffect(() => {
    if (estado !== 'listo') return;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: camaraTrasera ? 'environment' : 'user' },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (e) {
        console.error(e);
        setEstado('sin-camara');
      }
    })();
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [estado, camaraTrasera]);

  // ---- bucle de clasificacion: continuo, como el de predict_camera.py con OpenCV ----
  // Sin intervalo fijo: apenas termina una inferencia se toma el frame MAS RECIENTE del
  // video y se clasifica el siguiente. La tasa de cuadros medidos la pone el hardware
  // (los modelos V1 tardan decenas de ms en wasm); el video de fondo nunca se frena.
  useEffect(() => {
    if (estado !== 'listo') return;
    let activo = true;
    // anti-parpadeo: a cuadro por cuadro la especie puede oscilar en el borde de decision;
    // se muestra la mayoria de los ultimos 5 cuadros, igual de fresca pero estable
    const historial: Prediccion['especie'][] = [];
    (async () => {
      while (activo) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const pipeline = pipelineRef.current;
        if (!video || !canvas || !pipeline || pausadoRef.current || fotoRef.current
            || video.readyState < 2 || video.videoWidth === 0) {
          await new Promise((r) => setTimeout(r, 120));
          continue;
        }
        try {
          const m = modoRef.current;
          const frameworks: Framework[] = m === 'comparar' ? ['pytorch', 'tensorflow'] : [m];
          const preds: Prediccion[] = [];
          for (const fw of frameworks) preds.push(await pipeline.predecir(video, canvas, fw, true));
          // el anti-parpadeo se decide con el primer framework del modo activo
          historial.push(preds[0].especie);
          if (historial.length > 5) historial.shift();
          const conteo = new Map<string, number>();
          for (const e of historial) conteo.set(e, (conteo.get(e) ?? 0) + 1);
          const estable = [...conteo.entries()].sort((a, b) => b[1] - a[1])[0][0];
          // un cuadro suelto que contradice a la mayoria no pisa lo mostrado
          if (activo && estable === preds[0].especie) setPredicciones(preds);
        } catch (e) {
          console.error('inferencia fallo:', e);
          await new Promise((r) => setTimeout(r, 500));
        }
        // yield al renderer entre cuadros para que la UI y el video respiren
        await new Promise(requestAnimationFrame);
      }
    })();
    return () => { activo = false; };
  }, [estado]);

  const clasificarFoto = async (url: string, m: Modo) => {
    const canvas = canvasRef.current;
    const pipeline = pipelineRef.current;
    if (!canvas || !pipeline) return;
    const img = new Image();
    img.src = url;
    await img.decode();
    const frameworks: Framework[] = m === 'comparar' ? ['pytorch', 'tensorflow'] : [m];
    const preds: Prediccion[] = [];
    for (const fw of frameworks) preds.push(await pipeline.predecir(img, canvas, fw));
    setPredicciones(preds);
  };

  const elegirFoto = (e: React.ChangeEvent<HTMLInputElement>) => {
    const archivo = e.target.files?.[0];
    e.target.value = ''; // permite volver a elegir el mismo archivo
    if (!archivo) return;
    if (fotoRef.current) URL.revokeObjectURL(fotoRef.current);
    const url = URL.createObjectURL(archivo);
    setFoto(url);
    setPredicciones([]);
    void clasificarFoto(url, modoRef.current);
  };

  const volverACamara = () => {
    if (fotoRef.current) URL.revokeObjectURL(fotoRef.current);
    setFoto(null);
    setPredicciones([]);
  };

  const cambiarModo = (m: Modo) => {
    setModo(m);
    setPredicciones([]);
    // sobre una foto quieta el bucle no corre: reclasificar aca con el modo nuevo
    if (fotoRef.current) void clasificarFoto(fotoRef.current, m);
  };

  return (
    <IonApp>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Mascotas Live</IonTitle>
          <IonBadge slot="end" color="tertiary" className="badge-ondevice">ONNX on-device</IonBadge>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        {estado === 'cargando' && (
          <div className="estado-centrado">
            <IonSpinner name="crescent" />
            <p>{progreso}</p>
          </div>
        )}
        {estado === 'error' && <div className="estado-centrado"><p>{progreso}</p></div>}
        {estado === 'sin-camara' && (
          <div className="estado-centrado">
            <p>No hay acceso a la camara. Revisa el permiso de camara de la app.</p>
          </div>
        )}

        <div className="visor" style={{ display: estado === 'listo' ? 'block' : 'none' }}>
          <video
            ref={videoRef}
            playsInline
            muted
            className="video-preview"
            style={{ display: foto ? 'none' : 'block' }}
          />
          {foto && <img src={foto} alt="Foto elegida de la galeria" className="video-preview" />}

          <IonSegment
            value={modo}
            onIonChange={(e) => cambiarModo(e.detail.value as Modo)}
            className="selector-fw"
          >
            <IonSegmentButton value="pytorch"><IonLabel>PyTorch</IonLabel></IonSegmentButton>
            <IonSegmentButton value="tensorflow"><IonLabel>TensorFlow</IonLabel></IonSegmentButton>
            <IonSegmentButton value="comparar"><IonLabel>Comparar</IonLabel></IonSegmentButton>
          </IonSegment>

          {predicciones.length > 0 && (
            <div className={predicciones.length > 1 ? 'panel-doble' : 'panel-simple'}>
              {predicciones.map((p) => (
                <div className="overlay-resultado" key={p.framework}>
                  <div className="chip-fw">{CHIP[p.framework]}</div>
                  <div className="linea-especie">
                    <span className="emoji">{EMOJI[p.especie] ?? '?'}</span>
                    <span className="especie">{p.especie}</span>
                    <IonBadge color={p.especie === 'ninguno' ? 'medium' : 'success'}>
                      {(p.especieConfianza * 100).toFixed(0)}%
                    </IonBadge>
                  </div>

                  {p.raza && (
                    <div className="linea-raza">
                      {p.razaIdentificada ? (
                        <span>{p.raza.replace(/_/g, ' ')} ({(p.razaConfianza! * 100).toFixed(0)}%)</span>
                      ) : (
                        <span className="no-identificada">raza no identificada</span>
                      )}
                    </div>
                  )}

                  {predicciones.length === 1 && p.topRazas.length > 0 && (
                    <div className="top-razas">
                      {p.topRazas.map(([nombre, prob]) => (
                        <div key={nombre} className="fila-raza">
                          <span>{nombre.replace(/_/g, ' ')}</span>
                          <span className="prob">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="latencia">
                    {p.latenciaMs.toFixed(0)} ms &middot; {(1000 / p.latenciaMs).toFixed(1)} c/s
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="controles">
            {foto ? (
              <IonButton shape="round" onClick={volverACamara}>
                <IonIcon slot="start" icon={videocam} />
                Volver a la camara
              </IonButton>
            ) : (
              <>
                <IonButton shape="round" onClick={() => setPausado(!pausado)}>
                  <IonIcon slot="icon-only" icon={pausado ? play : pause} />
                </IonButton>
                <IonButton shape="round" fill="outline" onClick={() => setCamaraTrasera(!camaraTrasera)}>
                  <IonIcon slot="icon-only" icon={cameraReverse} />
                </IonButton>
              </>
            )}
            <IonButton shape="round" fill="outline" onClick={() => inputFotoRef.current?.click()}>
              <IonIcon slot="icon-only" icon={images} />
            </IonButton>
          </div>

          <input
            ref={inputFotoRef}
            type="file"
            accept="image/*"
            onChange={elegirFoto}
            style={{ display: 'none' }}
          />
        </div>
      </IonContent>
    </IonApp>
  );
};

export default App;
