# Altur Challenge — Detector Humano vs Sintético (HackMTY 2026)

Clasifica si el caller (canal 0) de una llamada de servicio al cliente bancario
es una persona real o un sistema autónomo (ASR → LLM → TTS).

## Idea central del diseño

El split train/val es **speaker-disjoint** y el juez usa voces nunca vistas.
Si el modelo aprende "voces", va a fallar. Por eso este pipeline apuesta a
**tres familias de señales que generalizan** a hablantes nuevos:

1. **Timing / turn-taking** (`features/timing_features.py`) — la señal más
   barata y más discriminante. Un pipeline ASR→LLM→TTS tiene latencias de
   respuesta más uniformes, casi no genera habla superpuesta genuina
   (backchannels tipo "ajá" mientras el otro habla), y reacciona "demasiado
   limpio" cuando el agente lo interrumpe.
2. **Acústica anti-spoofing** (`features/acoustic_features.py`) — jitter/shimmer,
   planitud espectral, MFCCs, respiraciones cortas dentro de los turnos, piso
   de ruido de fondo entre turnos.
3. **Lingüística vía ASR** (`features/linguistic_features.py`, opcional,
   `use_asr=True`) — cuando el agente pregunta algo sin sentido, un humano
   muestra confusión genuina ("¿cómo?", pausas), un LLM confabula fluido.

**Importante:** en evaluación (`POST /detect`) solo te dan el WAV, no el
`turns.json`. Por eso todo el pipeline —train e inferencia— usa un VAD propio
por energía (`features/turn_detection.py`) para derivar los turnos, en vez de
depender del `turns.json` "gold". Así no hay mismatch entre entrenamiento y
producción. Usa `src/evaluate_vad.py` para auditar qué tan bien tu VAD se
parece a los turns.json reales de train/val.

## Estructura

```
altur-challenge/
├── data/
│   ├── manifest.csv        # ya incluido con las filas que compartiste
│   ├── audio/<id>.wav       # <- copia aquí el contenido del zip
│   └── turns/<id>.json      # <- copia aquí (opcional, solo para auditoría)
├── src/
│   ├── features/
│   │   ├── turn_detection.py     # VAD propio por energía
│   │   ├── timing_features.py    # features de turn-taking
│   │   ├── acoustic_features.py  # features anti-spoofing
│   │   ├── linguistic_features.py# features via ASR (opcional)
│   │   └── feature_pipeline.py   # combina todo en un vector
│   ├── train.py             # entrena el clasificador
│   ├── evaluate_vad.py      # audita el VAD propio vs turns.json
│   └── api/server.py        # FastAPI con POST /detect
├── models/                  # aquí se guarda el .joblib entrenado
└── requirements.txt
```

## Cómo correrlo

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. descomprime el audio del reto
unzip altur-challenge-audio.zip -d data/audio/
# (y copia turns/*.json a data/turns/ si quieres auditar el VAD)

# 2. entrena
cd src
python train.py --data-dir ../data --out ../models/model.joblib

# 3. (opcional) audita el VAD contra los turns.json reales
python evaluate_vad.py --data-dir ../data

# 4. levanta el servidor de evaluación
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Probar el endpoint:
```bash
python - <<'PY'
import base64, requests
wav = base64.b64encode(open("../data/audio/call_04d682ac0cef.wav","rb").read()).decode()
r = requests.post("http://localhost:8000/detect", json={"audio_base64": wav})
print(r.json())
PY
```

## Con solo 12 llamadas de ejemplo (2 humanas, 10 sintéticas)

Ojo: el manifest que compartiste está MUY desbalanceado (10 synthetic / 2 human).
Con eso el modelo se puede ir a predecir "synthetic" siempre y aun así verse
bien en accuracy. Antes de confiar en el modelo:
- Revisa que el dataset real completo tenga mejor balance (probablemente sí,
  esto es solo una muestra que pegaste en el chat).
- Usa `class_weight`/`sample_weight` o balanced sampling si el desbalance persiste
  en el set completo.
- Mira siempre F1/AUC por clase, no solo accuracy global.

## Roadmap de mejoras (para diferenciarte en la demo)

- [ ] **Dashboard de explicabilidad**: por llamada, mostrar qué features
  pesaron más (SHAP values sobre el GradientBoostingClassifier). Para un caso
  de uso bancario, explicabilidad = tan importante como accuracy.
- [ ] **Calibración**: ya está con `CalibratedClassifierCV` (Platt scaling),
  el reto premia `confidence` honesto — verifica con un reliability diagram.
- [ ] **Robustez a speaker-disjoint**: si tienes tiempo, agrega augmentation
  (ruido, time-stretch leve, codecs de teléfono distintos) para simular
  condiciones no vistas.
- [ ] **Fine-tune de embeddings anti-spoofing** (wav2vec2-XLS-R con checkpoints
  de ASVspoof) si hay tiempo/GPU — mejor techo de accuracy que features hechas
  a mano, pero features a mano son más rápidas de defender ante el jurado.
- [ ] **Modo streaming**: actualizar `confidence` turno a turno en vez de
  esperar la llamada completa — más cercano a un caso de uso real de fraude
  en vivo en banca.
