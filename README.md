# Altur Challenge — Detector Humano vs Sintético (HackMTY 2026)

Clasifica si el caller (canal 0) de una llamada de servicio al cliente bancario
es una persona real o un sistema autónomo (ASR → LLM → TTS).

**Endpoint en producción:** `https://altur-challenge.onrender.com/detect`
> Nota: plan free de Render — se "duerme" tras ~15 min sin tráfico entrante.
> Mitigado con keep-alive automático (ver abajo), pero de todos modos corran
> `scripts/warmup.py` unos minutos antes de la evaluación en vivo.

## Warm-up antes de la evaluación

Dos capas, para no depender de una sola:

1. **Keep-alive automático** (`src/api/server.py`): mientras el proceso siga
   corriendo, se auto-pinguea `/health` cada 10 min (< 15 min del timeout de
   Render), usando `RENDER_EXTERNAL_URL` que Render define solo. No requiere
   ninguna acción manual una vez desplegado — el servicio deja de dormirse
   por completo.
2. **Warm-up manual** (`scripts/warmup.py`), por si el servicio se
   redesplegó hace poco o el keep-alive aún no alcanzó a activarse: manda
   llamadas reales a `/detect` (no solo `/health`) para calentar también el
   pipeline de features, y confirma que la latencia ya bajó a un rango
   razonable antes de que el juez se siente en la mesa.
   ```bash
   python scripts/warmup.py --url https://altur-challenge.onrender.com
   ```

## Contrato oficial del juez (confirmado por los organizadores)

El benchmark corre **desde la laptop de los jueces contra nuestro endpoint
desplegado**, **antes** de la explicación del proyecto, con **~100 llamadas**
del set oculto (mismo motor de voz que train/val, pero hablantes nuevos).

Request:
```json
{"call_id": "...", "audio_base64": "<base64 del WAV completo>", "sample_rate": 8000, "channels": 2}
```
`audio_base64` decodifica los bytes exactos de un WAV estéreo: canal 0 =
caller, canal 1 = agente.

Response esperada: HTTP 200 con
```json
{"is_synthetic": true, "confidence": 0.87}
```
`is_synthetic` es obligatorio; `confidence` es opcional pero se usa para
desempate y para medir calibración.

**Regla crítica:** un timeout, un status != 200, o una respuesta sin
`is_synthetic` booleano cuenta como respuesta incorrecta — igual que una
predicción equivocada. Por eso `api/server.py` nunca deja escapar una
excepción como error HTTP: cualquier fallo interno (JSON malformado, base64
corrupto, WAV inválido) cae a una respuesta de fallback con status 200.

Entrega en Devpost: **repositorio + URL del endpoint desplegado** (nada de
video/capturas obligatorias).

## Resultados (split val, speaker-disjoint)

| Métrica | Valor |
|---|---|
| Accuracy | 0.958 |
| AUC | 0.989 |
| Precision (synthetic) | 0.92 |
| Recall (synthetic) | 1.00 |
| Precision (human) | 1.00 |
| Recall (human) | 0.92 |

Entrenado con 282 llamadas de `train` (113 human / 169 synthetic), validado
contra 71 llamadas de `val` no vistas en entrenamiento.

**Features con más peso en el modelo** (importancia promedio del ensemble):
`spectral_flatness_std` y `spectral_flatness_mean` concentran ~79% de la
importancia total — la textura espectral de la voz (qué tan "plana"/sintética
suena frente a qué tan rica en armónicos naturales) es la señal más fuerte
en este dataset, seguida de la duración mínima de turno del caller y la
tasa de turnos por minuto.

## Idea central del diseño

El split train/val es **speaker-disjoint** y el juez usa voces nunca vistas.
Si el modelo aprende "voces", va a fallar. Por eso este pipeline apuesta a
**señales que generalizan** a hablantes nuevos, siguiendo las tres líneas que
el propio reto sugiere:

1. **Acústica** (`features/acoustic_features.py`) — jitter/shimmer, planitud
   espectral, MFCCs, respiraciones cortas dentro de los turnos, piso de ruido
   de fondo entre turnos. Es la familia de features con más peso en el modelo
   actual (ver tabla de arriba).
2. **Comportamiento conversacional / timing** (`features/timing_features.py`)
   — latencia de respuesta tras hablar el agente, habla superpuesta genuina
   (backchannels), y cómo reacciona el caller cuando el agente lo interrumpe.
   Un pipeline ASR→LLM→TTS tiende a tener latencias más uniformes y a
   reaccionar "demasiado limpio" ante una interrupción.
3. **Semántica vía ASR** (`features/linguistic_features.py`, opcional,
   `use_asr=True`) — cuando el agente pregunta algo sin sentido, un humano
   muestra confusión genuina ("¿cómo?", pausas), un LLM confabula fluido.
   No activada en el modelo actual (ver Roadmap).

**Importante:** en evaluación (`POST /detect`) solo se recibe el WAV, no el
`turns.json`. Por eso todo el pipeline —entrenamiento e inferencia— usa un VAD
propio por energía (`features/turn_detection.py`) para derivar los turnos,
en vez de depender del `turns.json` "gold" del dataset. Así no hay mismatch
entre lo que el modelo vio en entrenamiento y lo que ve en producción.
`src/evaluate_vad.py` audita qué tan bien este VAD se parece a los turns.json
reales de train/val.

## Estructura

```
altur-challenge/
├── data/
│   ├── manifest.csv
│   ├── audio/<id>.wav
│   └── turns/<id>.json      # usado solo para auditoría del VAD, no en train real
├── src/
│   ├── features/
│   │   ├── turn_detection.py     # VAD propio por energía
│   │   ├── timing_features.py    # features de turn-taking
│   │   ├── acoustic_features.py  # features anti-spoofing
│   │   ├── linguistic_features.py# features via ASR (opcional)
│   │   └── feature_pipeline.py   # combina todo en un vector (50 features)
│   ├── train.py             # entrena el clasificador (GradientBoosting + calibración)
│   ├── evaluate_vad.py      # audita el VAD propio vs turns.json
│   └── api/server.py        # FastAPI con POST /detect
├── models/model.joblib      # modelo entrenado, ya incluido en el repo
└── requirements.txt
```

## Cómo correrlo localmente

```bash
pip install -r requirements.txt

# entrenar (el modelo ya viene entrenado en models/model.joblib, este paso es
# solo si quieres reentrenar con más datos o cambios en las features)
cd src
python train.py --data-dir ../data --out ../models/model.joblib

# levantar el servidor
export ALTUR_MODEL_PATH=../models/model.joblib   # PowerShell: $env:ALTUR_MODEL_PATH="../models/model.joblib"
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Probar el endpoint (PowerShell):
```powershell
$bytes = [System.IO.File]::ReadAllBytes("..\data\audio\<id>.wav")
$b64 = [Convert]::ToBase64String($bytes)
$body = @{ audio_base64 = $b64 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method Post -Body $body -ContentType "application/json"
```

## Modelo

`GradientBoostingClassifier` envuelto en `CalibratedClassifierCV` (Platt
scaling), para que el `confidence` devuelto sea honesto — el reto lo usa
para desempatar y premiar sistemas bien calibrados, no solo el booleano
`is_synthetic`.

## Integración con el resto del equipo

> Sección viva — actualizar conforme se defina cómo encajan las piezas.

- **API (Gabi):** por definir si es la capa que reemplaza/envuelve
  `src/api/server.py`, o un componente aparte (frontend, orquestador, etc.).
  Mientras se aclara, el contrato de `/detect` (input/output) descrito arriba
  se mantiene fijo para no romper nada de lo ya hosteado en producción.
- **Modelo/dataset adicional (Julieta):** repo externo de detección de voz
  sintética a integrar con los datos de Altur. Dos formas posibles de
  integración, a decidir según qué tan compatible sea:
  1. Como **feature adicional**: se le pasa el audio, regresa una probabilidad,
     se agrega al vector de 50 features actuales antes de la capa de decisión.
  2. Como **segundo modelo en ensemble**: se promedian (o se combinan con un
     meta-clasificador) las probabilidades de ambos modelos.

## Roadmap (si sobra tiempo)

- [ ] Activar `use_asr=True` para las features semánticas (línea 3 del diseño).
- [ ] Dashboard de explicabilidad por llamada (qué features pesaron más).
- [ ] Reliability diagram para verificar la calibración del `confidence`.
- [ ] Fingerprinting de motor de síntesis (qué TTS generó la llamada).
- [ ] Correlación entre llamadas sintéticas (mismo origen/campaña de fraude).
