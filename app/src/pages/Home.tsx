import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import {
  IonBadge, IonButton, IonCard, IonCardContent, IonCardHeader, IonCardTitle, IonContent,
  IonHeader, IonIcon, IonInput, IonItem, IonLabel, IonList, IonModal, IonNote, IonPage,
  IonProgressBar, IonSegment, IonSegmentButton, IonSpinner, IonTitle, IonToolbar,
  useIonToast,
} from '@ionic/react';
import { camera, images, settingsOutline } from 'ionicons/icons';
import { useEffect, useRef, useState } from 'react';
import {
  clasificar, guardarUrlApi, obtenerUrlApi, verificarEstado,
  type Modo, type Prediccion,
} from '../api';
import './Home.css';

const EMOJI: Record<string, string> = { perro: '🐶', gato: '🐱', ninguno: '🚫' };

const Home: React.FC = () => {
  const [foto, setFoto] = useState<string | null>(null);
  const [resultados, setResultados] = useState<Record<string, Prediccion> | null>(null);
  // vista: 'demo' = los modelos de la presentacion (v1, con comparacion de frameworks);
  // 'pro' = los modelos con transfer learning
  const [vista, setVista] = useState<'demo' | 'pro'>('demo');
  const [modoDemo, setModoDemo] = useState<Exclude<Modo, 'pro'>>('pytorch');
  const [proDisponible, setProDisponible] = useState<boolean | null>(null);
  const [cargando, setCargando] = useState(false);
  const [conectado, setConectado] = useState<boolean | null>(null);
  const [urlApi, setUrlApi] = useState('');
  const [ajustesAbiertos, setAjustesAbiertos] = useState(false);
  const [mostrarToast] = useIonToast();
  const modal = useRef<HTMLIonModalElement>(null);

  useEffect(() => {
    obtenerUrlApi().then(setUrlApi);
    comprobarConexion();
  }, []);

  const comprobarConexion = async () => {
    try {
      const estado = await verificarEstado();
      setConectado(estado.ok);
      setProDisponible(estado.frameworks ? estado.frameworks.includes('pro') : null);
      if (estado.modo === 'demo') {
        mostrarToast({ message: 'Servidor en modo demo: resultados simulados', duration: 3000, color: 'warning' });
      }
    } catch {
      setConectado(false);
    }
  };

  const tomarFoto = async (origen: CameraSource) => {
    try {
      const imagen = await Camera.getPhoto({
        quality: 85,
        allowEditing: false,
        resultType: CameraResultType.Base64,
        source: origen,
      });
      if (!imagen.base64String) return;

      setFoto(`data:image/jpeg;base64,${imagen.base64String}`);
      setResultados(null);
      setCargando(true);
      try {
        const modo: Modo = vista === 'pro' ? 'pro' : modoDemo;
        setResultados(await clasificar(imagen.base64String, modo));
        setConectado(true);
      } catch (error) {
        setConectado(false);
        mostrarToast({
          message: `No se pudo clasificar: ${(error as Error).message}`,
          duration: 4000,
          color: 'danger',
        });
      } finally {
        setCargando(false);
      }
    } catch {
      // el usuario cancelo la camara: no es un error que valga la pena mostrar
    }
  };

  const guardarAjustes = async () => {
    await guardarUrlApi(urlApi);
    setAjustesAbiertos(false);
    await comprobarConexion();
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Clasificador de Mascotas</IonTitle>
          <IonButton slot="end" fill="clear" onClick={() => setAjustesAbiertos(true)}>
            <IonIcon icon={settingsOutline} />
          </IonButton>
        </IonToolbar>
      </IonHeader>

      <IonContent className="ion-padding">
        {conectado === false && (
          <IonCard color="danger">
            <IonCardContent>
              No hay conexion con el servidor. Verifica que <code>inference/server.py</code> este
              corriendo y que la URL sea correcta (boton de ajustes).
            </IonCardContent>
          </IonCard>
        )}

        <IonSegment
          value={vista}
          onIonChange={(e) => { setVista(e.detail.value as 'demo' | 'pro'); setResultados(null); }}
          className="selector-modo"
        >
          <IonSegmentButton value="demo"><IonLabel>Demo</IonLabel></IonSegmentButton>
          <IonSegmentButton value="pro"><IonLabel>Pro</IonLabel></IonSegmentButton>
        </IonSegment>

        {vista === 'demo' ? (
          <IonSegment
            value={modoDemo}
            onIonChange={(e) => { setModoDemo(e.detail.value as Exclude<Modo, 'pro'>); setResultados(null); }}
            className="selector-framework"
          >
            <IonSegmentButton value="pytorch"><IonLabel>PyTorch</IonLabel></IonSegmentButton>
            <IonSegmentButton value="tensorflow"><IonLabel>TensorFlow</IonLabel></IonSegmentButton>
            <IonSegmentButton value="comparar"><IonLabel>Comparar</IonLabel></IonSegmentButton>
          </IonSegment>
        ) : (
          <>
            <IonNote className="nota-pro">
              Transfer learning: mas razas, confianza calibrada y entrenamiento pensado para
              fotos reales de camara.
            </IonNote>
            {proDisponible === false && (
              <IonCard color="warning">
                <IonCardContent>
                  El servidor todavia no tiene los modelos pro cargados (faltan los
                  <code> models/*_pro_pytorch.pt</code>, se bajan de la EC2 cuando termine el
                  entrenamiento). Mientras tanto las fotos en este modo van a fallar.
                </IonCardContent>
              </IonCard>
            )}
          </>
        )}

        <div className="botones-captura">
          <IonButton expand="block" size="large" onClick={() => tomarFoto(CameraSource.Camera)}>
            <IonIcon slot="start" icon={camera} />
            Tomar foto
          </IonButton>
          <IonButton expand="block" fill="outline" onClick={() => tomarFoto(CameraSource.Photos)}>
            <IonIcon slot="start" icon={images} />
            Elegir de la galeria
          </IonButton>
        </div>

        {foto && (
          <IonCard>
            <img src={foto} alt="Foto capturada" className="foto-preview" />
            {cargando && <IonProgressBar type="indeterminate" />}
          </IonCard>
        )}

        {cargando && (
          <div className="estado-centrado">
            <IonSpinner name="crescent" />
            <p>Analizando...</p>
          </div>
        )}

        {resultados && !cargando && Object.entries(resultados).map(([framework, resultado]) => (
          <IonCard key={framework}>
            <IonCardHeader>
              {(framework === 'pro' || Object.keys(resultados).length > 1) && (
                <IonBadge
                  color={framework === 'pro' ? 'tertiary' : 'medium'}
                  className="badge-framework"
                >
                  {framework === 'pro' ? 'PRO' : framework}
                </IonBadge>
              )}
              <IonCardTitle>
                {EMOJI[resultado.especie] ?? '?'} {resultado.resumen}
              </IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <IonItem lines="none">
                <IonLabel>Especie</IonLabel>
                <IonBadge slot="end" color={resultado.especie === 'ninguno' ? 'medium' : 'success'}>
                  {resultado.especie} {(resultado.especie_confianza * 100).toFixed(0)}%
                </IonBadge>
              </IonItem>

              {!resultado.raza_identificada && resultado.raza && (
                <IonNote color="warning" className="nota-umbral">
                  La confianza no alcanza el umbral calibrado, asi que se reporta como raza no
                  identificada en vez de arriesgar una respuesta incorrecta.
                </IonNote>
              )}

              {resultado.top_razas.length > 0 && (
                <>
                  <p className="titulo-lista">Razas mas probables</p>
                  <IonList>
                    {resultado.top_razas.map(([nombre, prob]) => (
                      <IonItem key={nombre}>
                        <IonLabel>{nombre.replace(/_/g, ' ')}</IonLabel>
                        <IonBadge slot="end">{(prob * 100).toFixed(1)}%</IonBadge>
                      </IonItem>
                    ))}
                  </IonList>
                </>
              )}
            </IonCardContent>
          </IonCard>
        ))}

        <IonModal ref={modal} isOpen={ajustesAbiertos} onDidDismiss={() => setAjustesAbiertos(false)}>
          <IonHeader>
            <IonToolbar>
              <IonTitle>Ajustes</IonTitle>
              <IonButton slot="end" fill="clear" onClick={guardarAjustes}>Guardar</IonButton>
            </IonToolbar>
          </IonHeader>
          <IonContent className="ion-padding">
            <IonItem>
              <IonLabel position="stacked">URL del servidor</IonLabel>
              <IonInput
                value={urlApi}
                placeholder="http://192.168.1.100:8000"
                onIonInput={(e) => setUrlApi(e.detail.value ?? '')}
              />
            </IonItem>
            <IonNote className="ayuda-url">
              Es la IP de la computadora donde corre <code>inference/server.py</code>, en la misma
              red WiFi que el celular. El servidor la imprime al arrancar.
            </IonNote>
          </IonContent>
        </IonModal>
      </IonContent>
    </IonPage>
  );
};

export default Home;
