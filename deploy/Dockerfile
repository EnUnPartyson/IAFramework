# Imagen del servidor de inferencia (CPU) para Hugging Face Spaces o cualquier host Docker.
#
# No copia codigo: clona el repo publico de GitHub, que ya incluye los pesos en models/.
# Para actualizar modelos (ej. cuando esten los pro): rebuild (en HF: Settings -> Factory rebuild).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces ejecuta el contenedor con el usuario 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

ARG REPO_URL=https://github.com/EnUnPartyson/IAFramework.git
# rama o tag a servir; "v1-presentacion" fija la version exacta de la presentacion
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO_URL} . && rm -rf .git

RUN pip install --no-cache-dir -r requirements-inference-cpu.txt

USER user
EXPOSE 7860
# "ambos" carga PyTorch + TensorFlow (modo Demo con comparacion) y el pro se suma solo
# cuando sus pesos existan en models/
CMD ["python", "inference/server.py", "--host", "0.0.0.0", "--port", "7860", "--frameworks", "ambos"]
