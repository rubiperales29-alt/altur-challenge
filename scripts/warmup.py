"""
Calienta el endpoint /detect desplegado justo antes de la evaluación en vivo.

El keep-alive de src/api/server.py evita que Render duerma el servicio
mientras el proceso sigue corriendo, pero conviene correr esto de todos
modos unos minutos antes de que llegue el juez a la mesa: confirma que el
servicio responde y que el pipeline de features ya está caliente (primeras
llamadas después de un despliegue nuevo suelen ser más lentas).

Solo usa la librería estándar de Python.

Uso:
    python scripts/warmup.py --url https://altur-challenge.onrender.com
"""
from __future__ import annotations
import argparse
import base64
import csv
import json
import os
import time
import urllib.request


def _read_sample_call(data_dir: str) -> tuple[str, bytes]:
    manifest_path = os.path.join(data_dir, "manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["split"] == "val"), rows[0])
    call_id = row["anon_id"]
    wav_path = os.path.join(data_dir, "audio", f"{call_id}.wav")
    with open(wav_path, "rb") as f:
        return call_id, f.read()


def _ping_health(base_url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  /health falló: {e}")
        return False


def _call_detect(base_url: str, call_id: str, wav_bytes: bytes, timeout: float) -> float | None:
    payload = {
        "call_id": call_id,
        "audio_base64": base64.b64encode(wav_bytes).decode(),
        "sample_rate": 8000,
        "channels": 2,
    }
    req = urllib.request.Request(
        f"{base_url}/detect",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            dt = (time.time() - t0) * 1000
            ok = resp.status == 200 and isinstance(body.get("is_synthetic"), bool)
            print(f"  /detect -> status={resp.status} body={body} ({dt:.0f}ms) {'OK' if ok else 'FORMATO INESPERADO'}")
            return dt
    except Exception as e:
        print(f"  /detect falló: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Calienta el endpoint /detect antes de la evaluación en vivo.")
    ap.add_argument("--url", required=True, help="Base URL del endpoint, sin /detect (ej: https://altur-challenge.onrender.com)")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--rounds", type=int, default=3, help="Máximo de llamadas reales a /detect")
    ap.add_argument("--timeout", type=float, default=35.0, help="Timeout por request en segundos")
    ap.add_argument("--target-ms", type=float, default=15000, help="Latencia objetivo para considerarlo 'caliente'")
    args = ap.parse_args()

    base_url = args.url.rstrip("/")
    call_id, wav_bytes = _read_sample_call(args.data_dir)
    print(f"Usando llamada de prueba: {call_id} ({len(wav_bytes)} bytes)\n")

    print("1) Despertando el servicio (/health)...")
    for attempt in range(1, 6):
        if _ping_health(base_url, timeout=args.timeout):
            print("   servicio respondiendo.\n")
            break
        print(f"   intento {attempt}/5, reintentando en 10s...")
        time.sleep(10)
    else:
        print("   no respondió /health tras varios intentos — revisa la URL o el dashboard de Render.")
        return

    print(f"2) Calentando el pipeline con hasta {args.rounds} llamadas reales a /detect...")
    warm = False
    for i in range(1, args.rounds + 1):
        print(f" ronda {i}/{args.rounds}:")
        dt = _call_detect(base_url, call_id, wav_bytes, timeout=args.timeout)
        if dt is not None and dt <= args.target_ms:
            warm = True
            break

    if warm:
        print(f"\nListo — el endpoint responde en menos de {args.target_ms:.0f}ms. Puede empezar la evaluación.")
    else:
        print(f"\nAún no baja de {args.target_ms:.0f}ms o falló — corre el script otra vez o revisa el servidor antes de que lleguen los jueces.")


if __name__ == "__main__":
    main()
