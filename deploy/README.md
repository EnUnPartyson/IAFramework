# Servidor en la nube (app disponible en cualquier momento)

La app le pega a `inference/server.py`. Corriendo el server en la laptop solo funciona en tu
WiFi y mientras la laptop este prendida; con esta carpeta lo subis a un host y la app anda
desde cualquier lado, siempre.

## Opción recomendada: Hugging Face Spaces (gratis)

CPU de 2 nucleos y 16 GB de RAM gratis, con HTTPS incluido. Ideal para este proyecto: la
inferencia en CPU tarda bien poco (los modelos son chicos) y no gasta créditos de AWS.

1. Cuenta gratuita en https://huggingface.co
2. **New Space** (botón en tu perfil):
   - Space name: `clasificador-mascotas` (o el que quieras)
   - SDK: **Docker** → template "Blank"
   - Visibility: **Public** (una Space privada exige token en cada request y la app no lo maneja)
3. En la Space: **Files → Add file → Upload files** y subí el `Dockerfile` de esta carpeta
   (tiene que llamarse exactamente `Dockerfile`, en la raíz de la Space).
4. La Space se construye sola (~5-10 min la primera vez: clona este repo con los pesos
   incluidos e instala las dependencias de CPU). Cuando diga "Running", tu URL es:

       https://<tu-usuario>-clasificador-mascotas.hf.space

5. En la app: Ajustes (engranaje) → URL del servidor → pegá esa URL → Guardar.
   Probá abrirla en el navegador del celular: tiene que aparecer la página de estado.

**Para actualizar los modelos** (ej. cuando bajes los pro de la EC2 y los pushees al repo):
Space → Settings → **Factory rebuild**. Vuelve a clonar el repo y levanta con los pesos nuevos.

**Dormida por inactividad:** el plan gratuito duerme la Space tras ~48 h sin tráfico y la
despierta sola con el primer request (tarda 1-2 min). Para la presentación: abrí la URL unos
minutos antes y queda caliente.

**Servir exactamente la versión de la presentación:** editá el `Dockerfile` en la Space y
cambiá `ARG REF=main` por `ARG REF=v1-presentacion`.

## Alternativa: EC2 chica siempre encendida (usa créditos)

Si preferís quedarte en AWS: una **t3.medium** (4 GB RAM, ~$0.05/h ≈ $33/mes de créditos)
alcanza para inferencia en CPU.

```bash
# en la instancia (Ubuntu), puerto 8000 abierto en el security group:
git clone https://github.com/EnUnPartyson/IAFramework.git && cd IAFramework
python3 -m venv venv-inf && venv-inf/bin/pip install -r requirements-inference-cpu.txt
tmux new -s api
venv-inf/bin/python inference/server.py --frameworks ambos
```

En la app va `http://<IP-elastica>:8000` (asigná una Elastic IP para que no cambie al
reiniciar). Contras frente a Spaces: cuesta créditos, es HTTP plano y hay que administrarla.

## Probar la imagen localmente (opcional, requiere Docker)

```bash
docker build -f deploy/Dockerfile -t mascotas-api deploy/
docker run --rm -p 8000:7860 mascotas-api
# en otra terminal: curl http://localhost:8000/health
```
