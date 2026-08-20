/**
 * Mascotas Live: camara en vivo + inferencia ONNX on-device. Sin servidor, sin red:
 * los modelos V1 van empaquetados y clasifican el video continuamente, como hace
 * inference/predict_camera.py en la laptop con OpenCV.
 */
import {
  IonApp, IonBadge, IonButton, IonContent, IonHeader, IonIcon, IonSpinner,
  IonTitle, IonToolbar, setupIonicReact,
} from '@ionic/react';
import { cameraReverse, pause, play } from 'ionicons/icons';
import { useEffect, useRef, useState } from 'react';
import { PipelineOnnx, type Prediccion } from './pipeline';

import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import './App.css';

setupIonicReact();

const EMOJI: Record<string, string> = { perro: '\u{1F436}', gato: '\u{1F431}', ninguno: '\u{1F6AB}' };
// clasificar cada ~400ms: fluido para el ojo sin saturar el CPU del telefono
const INTERVALO_MS = 400;

type Estado = 'cargando' | 'listo' | 'sin-camara' | 'error';

const App: React.FC = () => {
  const [estado, setEstado] = useState<Estado>('cargando');
  const [progreso, setProgreso] = useState('Iniciando...');
  const [prediccion, setPrediccion] = useState<Prediccion | null>(null);
  const [pausado, setPausado] = useState(false);
  const [camaraTrasera, setCamaraTrasera] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pipelineRef = useRef<PipelineOnnx | null>(null);
  const pausadoRef = useRef(false);
  const corriendoRef = useRef(false);

  pausadoRef.current = pausado;

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

  // ---- bucle de clasificacion ----
  useEffect(() => {
    if (estado !== 'listo') return;
    const timer = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const pipeline = pipelineRef.current;
      // corriendoRef evita encolar inferencias si una tarda mas que el intervalo
      if (!video || !canvas || !pipeline || pausadoRef.current || corriendoRef.current) return;
      if (video.readyState < 2 || video.videoWidth === 0) return;
      corriendoRef.current = true;
      try {
        setPrediccion(await pipeline.predecir(video, canvas));
      } catch (e) {
        console.error('inferencia fallo:', e);
      } finally {
        corriendoRef.current = false;
      }
    }, INTERVALO_MS);
    return () => clearInterval(timer);
  }, [estado]);

  const p = prediccion;
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
          <video ref={videoRef} playsInline muted className="video-preview" />

          {p && (
            <div className="overlay-resultado">
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

              {p.topRazas.length > 0 && (
                <div className="top-razas">
                  {p.topRazas.map(([nombre, prob]) => (
                    <div key={nombre} className="fila-raza">
                      <span>{nombre.replace(/_/g, ' ')}</span>
                      <span className="prob">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="latencia">{p.latenciaMs.toFixed(0)} ms</div>
            </div>
          )}

          <div className="controles">
            <IonButton shape="round" onClick={() => setPausado(!pausado)}>
              <IonIcon slot="icon-only" icon={pausado ? play : pause} />
            </IonButton>
            <IonButton shape="round" fill="outline" onClick={() => setCamaraTrasera(!camaraTrasera)}>
              <IonIcon slot="icon-only" icon={cameraReverse} />
            </IonButton>
          </div>
        </div>
      </IonContent>
    </IonApp>
  );
};

export default App;
