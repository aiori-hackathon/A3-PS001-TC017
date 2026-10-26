from fastapi.encoders import jsonable_encoder
import math

from datetime import datetime, timezone
from math import isfinite
from threading import Lock
from typing import Any
from backend.event_aggregator import AndroidEventAggregator


from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import ranscope_m_unified_backend as model_pipeline
import io
import json
import tempfile
import os
import pandas as pd


app = FastAPI(
    title="RANSCOPE-M API",
    description="Telemetry ingestion and ransomware detection API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_WEIGHTS = {
    "MPVG": 0.18,
    "SFIM": 0.26,
    "EAP": 0.31,
    "CAIT": 0.25,
}
DETECTION_THRESHOLD = 0.50

latest_telemetry: dict[str, Any] | None = None
latest_detection: dict[str, Any] | None = None
latest_android_window: dict[str, Any] | None = None
# Bounded in-memory history. It survives page refreshes but resets on backend restart.
detection_history: list[dict[str, Any]] = []
MAX_DETECTION_HISTORY = 500
websocket_clients: set[WebSocket] = set()
android_aggregator = AndroidEventAggregator()

# Retain the most recently received individual Android event.
latest_android_event: dict[str, Any] | None = None

detector = None
model_load_error: str | None = None
detection_lock = Lock()


class TelemetryRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    telemetry: dict[str, Any]


class DetectionRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    window: dict[str, Any]


def required_feature_columns() -> list[str]:
    """Return every unique feature required by all four trained models."""
    return sorted({
        column
        for columns in model_pipeline.LAYER_COLUMNS.values()
        for column in columns
    })


REQUIRED_FEATURES = required_feature_columns()


@app.on_event("startup")
async def load_detection_models():
    """Load the pretrained artifact once when the API starts."""
    global detector, model_load_error

    try:
        for layer in model_pipeline.FILES:
            model_pipeline.RUNS.setdefault(layer, {})

        detector = model_pipeline.RansomwareEarlyWarning()
        model_load_error = None
        print("RANSCOPE-M: all four pretrained models loaded.")
    except Exception as exc:
        detector = None
        model_load_error = str(exc)
        print(f"RANSCOPE-M model loading failed: {exc}")


async def broadcast(payload: dict[str, Any]) -> None:
    """Broadcast an update to connected dashboard clients."""
    disconnected = set()

    for client in websocket_clients.copy():
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.add(client)

    websocket_clients.difference_update(disconnected)


def validate_window(window: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or non-numeric model feature windows."""
    missing = [
        name for name in REQUIRED_FEATURES
        if name not in window
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Incomplete feature window. No model inference was run."
                ),
                "missing_features": missing,
                "required_feature_count": len(REQUIRED_FEATURES),
            },
        )

    validated = dict(window)

    for name in REQUIRED_FEATURES:
        value = validated[name]

        if isinstance(value, bool):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be numeric, not boolean.",
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be numeric.",
            )

        if not isfinite(numeric_value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be finite.",
            )

        validated[name] = numeric_value

    return validated


@app.get("/")
async def root():
    return {
        "project": "RANSCOPE-M",
        "message": "Backend is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ranscope-m-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def api_status():
    return {
        "backend": "online",
        "telemetry_received": latest_telemetry is not None,
        "detection_available": latest_detection is not None,
        "connected_websocket_clients": len(websocket_clients),
        "model_inference": (
            "ready" if detector is not None else "unavailable"
        ),
        "model_load_error": model_load_error,
        "required_feature_count": len(REQUIRED_FEATURES),
        "model_weights": MODEL_WEIGHTS,
        "threshold": DETECTION_THRESHOLD,
    }


@app.post("/api/telemetry")
async def receive_telemetry(request: TelemetryRequest):
    """Accept and broadcast raw telemetry without inventing a prediction."""
    global latest_telemetry

    latest_telemetry = {
        "device_id": request.device_id,
        "timestamp": request.timestamp.isoformat(),
        "telemetry": request.telemetry,
    }

    event = {
        "event": "telemetry_update",
        "telemetry": latest_telemetry,
        "detection": latest_detection,
    }

    await broadcast(event)

    return {
        "accepted": True,
        "telemetry": latest_telemetry,
        "detection": None,
        "message": (
            "Telemetry received. Submit a complete feature window "
            "to /api/detections/run for model inference."
        ),
    }



@app.post("/api/telemetry/window")
async def receive_android_window(request: TelemetryRequest):
    """Store an Android window and run available pretrained layers."""
    global latest_android_window, latest_detection, latest_telemetry

    telemetry = request.telemetry

    if telemetry.get("event") != "android_window_complete":
        raise HTTPException(
            status_code=422,
            detail="Expected event='android_window_complete'.",
        )

    android_features = telemetry.get("feature_window")
    if not isinstance(android_features, dict):
        raise HTTPException(
            status_code=422,
            detail="feature_window must be a JSON object.",
        )

    latest_android_window = {
        "device_id": request.device_id,
        "timestamp": request.timestamp.isoformat(),
        "telemetry": telemetry,
    }

    await broadcast({
        "event": "android_window_update",
        "android_window": latest_android_window,
    })

    if detector is None:
        return {
            "accepted": True,
            "android_window": latest_android_window,
            "inference_run": False,
            "message": "Android window stored, but pretrained models are unavailable.",
        }

    # Translate only features whose meanings match the trained schema.
    # Do not invent permission or cryptographic API counts.
    def number(*names, default=0.0):
        for name in names:
            value = android_features.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    return float(value)
        return float(default)

    features = {
        "run_id": telemetry.get("run_id", "__android_live__"),
        "window_start_ms": number("window_start_ms"),
        "window_end_ms": number("window_end_ms"),
    }

    # Android file-monitor events -> shared file activity features.
    features.update({
        "create_count": number("sfim_create_count", "create_count"),
        "modify_count": number("sfim_modify_count", "modify_count"),
        "close_write_count": number("sfim_close_write_count", "close_write_count"),
        "delete_count": number("sfim_unlink_count", "delete_count"),
        "moved_from_count": number("sfim_moved_from_count", "moved_from_count"),
        "moved_to_count": number("sfim_moved_to_count", "moved_to_count"),
        "SFIM_file_read_open_count": number("sfim_read_count", "SFIM_file_read_open_count"),
        "SFIM_file_write_open_count": number("sfim_write_count", "SFIM_file_write_open_count"),
        "SFIM_file_delete_count": number("sfim_unlink_count", "SFIM_file_delete_count"),
        "SFIM_file_rename_count": number("sfim_rename_count", "SFIM_file_rename_count"),
        "file_event_count": number("sfim_event_count"),
        "unique_path_count": number("sfim_unique_path_count", "unique_path_count"),
        "avg_depth": number("sfim_mean_depth", "avg_depth"),
        "avg_entropy": number("sfim_mean_write_entropy", "sfim_mean_dir_entropy", "avg_entropy"),
        "max_entropy": number("eap_H", "max_entropy", default=0.0),
        "entropy_observation_count": number("eap_bucket_count", "entropy_observation_count"),
    })

    # Extension counters are only supplied if Android actually reports them.
    for ext in ("txt", "csv", "json", "pdf", "jpg", "png", "docx", "md", "log"):
        features[f"{ext}_count"] = number(f"{ext}_count")

    # CAIT API names must match the trained schema. Only map explicitly
    # reported matching counters; never turn a generic file event into an API call.
    for api_name in model_pipeline.CAIT_APIS:
        if api_name in android_features:
            features[api_name] = number(api_name)
        else:
            android_name = "cait_" + api_name.removeprefix("CAIT_").removesuffix("_count")
            if android_name in android_features:
                features[api_name] = number(android_name)

    # Permission and cryptographic API counts remain unavailable unless
    # Android instrumentation provides the exact corresponding measurements.
    if "MPVG_checkSelfPermission_count" in android_features:
        features["MPVG_checkSelfPermission_count"] = number(
            "MPVG_checkSelfPermission_count"
        )
    if "MPVG_requestPermissions_count" in android_features:
        features["MPVG_requestPermissions_count"] = number(
            "MPVG_requestPermissions_count"
        )
    if "EAP_cipher_doFinal_count" in android_features:
        features["EAP_cipher_doFinal_count"] = number(
            "EAP_cipher_doFinal_count"
        )

    # The detector itself determines which layers have all required columns.
    # Missing columns are intentionally not fabricated.
    available_layers = [
        layer for layer, columns in model_pipeline.LAYER_COLUMNS.items()
        if all(
            column in features
            for column in dict.fromkeys(model_pipeline.META + columns)
        )
    ]

    if not available_layers:
        timestamp = request.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        latest_telemetry = {
            "device_id": request.device_id,
            "timestamp": timestamp.isoformat(),
            "telemetry": features,
        }

        await broadcast({
            "event": "telemetry_update",
            "telemetry": latest_telemetry,
            "detection": latest_detection,
            "android_window": latest_android_window,
        })

        return {
            "accepted": True,
            "android_window": latest_android_window,
            "inference_run": False,
            "available_layers": [],
            "missing_features": {
                layer: [
                    col for col in columns
                    if col not in features
                ]
                for layer, columns in model_pipeline.LAYER_COLUMNS.items()
            },
            "message": (
                "Android window stored and telemetry updated. "
                "No model layer has a genuine usable signal yet; "
                "no new prediction was generated."
            ),
        }

    timestamp = request.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    try:
        with detection_lock:
            output = detector.push(features)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Android model inference failed: {type(exc).__name__}: {exc}",
        )

    ran_layers = list(output["p_layers"].keys())
    final_probability = float(output["p_final"])
    verdict = (
        "MALICIOUS"
        if final_probability >= DETECTION_THRESHOLD
        else "BENIGN"
    )

    latest_telemetry = {
        "device_id": request.device_id,
        "timestamp": timestamp.isoformat(),
        "telemetry": features,
    }

    latest_detection = {
        "device_id": request.device_id,
        "timestamp": timestamp.isoformat(),
        "model_outputs": {
            name.lower(): float(probability)
            for name, probability in output["p_layers"].items()
        },
        "model_weights": {
            name.lower(): weight
            for name, weight in MODEL_WEIGHTS.items()
        },
        "active_models": ran_layers,
        "inactive_models": [
            layer for layer in MODEL_WEIGHTS if layer not in ran_layers
        ],
        "final_probability": round(final_probability, 6),
        "risk_level": verdict,
        "threshold": DETECTION_THRESHOLD,
        "alert": bool(output["alert"]),
        "window": int(output["window"]),
        "inference_status": "real_model_inference",
    }
    detection_history.append(dict(latest_detection))
    if len(detection_history) > MAX_DETECTION_HISTORY:
        del detection_history[:-MAX_DETECTION_HISTORY]

    await broadcast({
        "event": "detection_update",
        "telemetry": latest_telemetry,
        "detection": latest_detection,
    })

    return {
        "accepted": True,
        "android_window": latest_android_window,
        "inference_run": True,
        "available_layers": ran_layers,
        "detection": latest_detection,
        "message": "Android window processed by the available pretrained layers.",
    }




# NEW ENDPOINT: individual Android/Frida monitoring events

@app.post("/api/telemetry/event")
async def receive_telemetry_event(request: dict[str, Any]):
    """Receive an Android event and update the in-memory feature aggregator."""
    global latest_telemetry, latest_android_event

    device_id = request.get("device_id", "ranscope-android")
    telemetry_event = request.get("telemetry", {})
    raw_timestamp = request.get("timestamp")

    if not isinstance(device_id, str) or not device_id.strip():
        raise HTTPException(
            status_code=422,
            detail="device_id must be a non-empty string.",
        )

    if not isinstance(telemetry_event, dict):
        raise HTTPException(
            status_code=422,
            detail="telemetry must contain a JSON object.",
        )

    try:
        if raw_timestamp is None:
            timestamp_text = datetime.now(timezone.utc).isoformat()
        elif isinstance(raw_timestamp, (int, float)) and not isinstance(
            raw_timestamp, bool
        ):
            timestamp_text = datetime.fromtimestamp(
                raw_timestamp, timezone.utc
            ).isoformat()
        elif isinstance(raw_timestamp, str):
            parsed_timestamp = datetime.fromisoformat(
                raw_timestamp.replace("Z", "+00:00")
            )
            if parsed_timestamp.tzinfo is None:
                parsed_timestamp = parsed_timestamp.replace(
                    tzinfo=timezone.utc
                )
            timestamp_text = parsed_timestamp.isoformat()
        else:
            raise ValueError("Unsupported timestamp format.")
    except (ValueError, OverflowError, OSError):
        raise HTTPException(
            status_code=422,
            detail="timestamp must be a valid ISO or Unix timestamp.",
        )

    # Add the actual observed event to the running aggregator.
    accepted = android_aggregator.add_event(telemetry_event)

    if not accepted:
        raise HTTPException(
            status_code=422,
            detail="Event must contain valid layer and event fields.",
        )

    snapshot = android_aggregator.snapshot()

    latest_android_event = {
        "device_id": device_id,
        "timestamp": timestamp_text,
        "telemetry": telemetry_event,
    }

    latest_telemetry = {
        "device_id": device_id,
        "timestamp": timestamp_text,
        "telemetry": telemetry_event,
        "aggregation": snapshot,
    }

    await broadcast({
        "event": "telemetry_update",
        "telemetry": latest_telemetry,
        "detection": latest_detection,
    })

    return {
        "accepted": True,
        "device_id": device_id,
        "event_type": telemetry_event.get("event", "unknown"),
        "layer": telemetry_event.get("layer", "unknown"),
        "aggregation": snapshot,
        "inference_run": False,
        "message": (
            "Event aggregated. Model inference remains disabled "
            "until all required features are genuinely available."
        ),
    }



@app.post("/api/detections/run")
async def run_detection(request: DetectionRequest):
    """Run the actual pretrained models on one complete telemetry window."""
    global latest_detection, latest_telemetry

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "The pretrained models are not available.",
                "reason": model_load_error,
            },
        )

    features = validate_window(request.window)

    timestamp = request.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    timestamp_text = timestamp.isoformat()

    # The detector uses shared per-layer RUNS state internally.
    # Serialize pushes to prevent concurrent requests from mixing that state.
    try:
        with detection_lock:
            output = detector.push(features)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        )

    layer_probabilities = {
        name: float(output["p_layers"][name])
        for name in MODEL_WEIGHTS
    }

    final_probability = float(output["p_final"])
    verdict = (
        "MALICIOUS"
        if final_probability >= DETECTION_THRESHOLD
        else "BENIGN"
    )

    latest_telemetry = {
        "device_id": request.device_id,
        "timestamp": timestamp_text,
        "telemetry": features,
    }

    latest_detection = {
        "device_id": request.device_id,
        "timestamp": timestamp_text,
        "model_outputs": {
            name.lower(): probability
            for name, probability in layer_probabilities.items()
        },
        "model_weights": {
            name.lower(): weight
            for name, weight in MODEL_WEIGHTS.items()
        },
        "final_probability": round(final_probability, 6),
        "risk_level": verdict,
        "threshold": DETECTION_THRESHOLD,
        "alert": bool(output["alert"]),
        "window": int(output["window"]),
        "inference_status": "real_model_inference",
    }
    detection_history.append(dict(latest_detection))
    if len(detection_history) > MAX_DETECTION_HISTORY:
        del detection_history[:-MAX_DETECTION_HISTORY]

    await broadcast({
        "event": "detection_update",
        "telemetry": latest_telemetry,
        "detection": latest_detection,
    })

    return {
        "accepted": True,
        "telemetry": latest_telemetry,
        "detection": latest_detection,
        "message": "Inference completed using the four pretrained models.",
    }


@app.post("/api/csv-analysis")
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """Analyze a unified telemetry CSV without changing live monitoring state."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")

    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="CSV exceeds the 15 MB upload limit."
        )

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {exc}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no data rows.")

    required_columns = (
        ["run_id", "window_start_ms", "window_end_ms"]
        + required_feature_columns()
    )

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV is missing required columns.",
                "missing_columns": missing,
            },
        )

    if len(df) > 10000:
        raise HTTPException(
            status_code=413,
            detail="CSV exceeds the 10,000-row limit."
        )

    if df["run_id"].isna().any():
        raise HTTPException(
            status_code=422,
            detail="run_id cannot contain empty values."
        )

    df["run_id"] = df["run_id"].astype(str)

    numeric_columns = (
        ["window_start_ms", "window_end_ms"]
        + required_feature_columns()
    )

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().any() or not df[col].map(isfinite).all():
            raise HTTPException(
                status_code=422,
                detail=f"Column '{col}' must contain finite numeric values."
            )

    if (df["window_end_ms"] < df["window_start_ms"]).any():
        raise HTTPException(
            status_code=422,
            detail="window_end_ms must be greater than or equal to window_start_ms."
        )

    if df.duplicated(["run_id", "window_start_ms"]).any():
        raise HTTPException(
            status_code=422,
            detail="Duplicate run_id and window_start_ms combinations found."
        )

    df = df.sort_values(
        ["run_id", "window_start_ms"]
    ).reset_index(drop=True)

    has_labels = "label" in df.columns and df["label"].notna().any()

    if has_labels:
        labels = df["label"].astype(str).str.strip().str.lower()
        aliases = {
            "1": "malicious",
            "0": "benign",
            "true": "malicious",
            "false": "benign",
        }
        labels = labels.map(lambda x: aliases.get(x, x))

        if labels.isna().any() or not set(labels.unique()).issubset(
            {"benign", "malicious"}
        ):
            raise HTTPException(
                status_code=422,
                detail="Labels must be benign/malicious or 0/1."
            )

        df["label"] = labels

    temp_path = None

    try:
        # Prevent concurrent live inference while the pipeline uses global RUNS.
        with detection_lock:
            previous_runs = {
                layer: model_pipeline.RUNS.get(layer)
                for layer in model_pipeline.FILES
            }

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".xlsx", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                result = model_pipeline.test_dataset(
                    df,
                    ensemble=model_pipeline.WeightedEnsemble(
                        weights=MODEL_WEIGHTS,
                        threshold=DETECTION_THRESHOLD,
                    ),
                    save_to=temp_path,
                    per_window=True,
                    parallel=False,
                    download=False,
                    export_excel=False,

                )

            finally:
                # Restore the original live-monitoring state.
                for layer, saved_runs in previous_runs.items():
                    if saved_runs is None:
                        model_pipeline.RUNS.pop(layer, None)
                    else:
                        model_pipeline.RUNS[layer] = saved_runs

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
           status_code=422,
           detail=f"CSV inference failed: {type(exc).__name__}: {exc}"
    )

    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    run_df = result["run_df"]
    window_df = result.get("win_df")
    metrics_df = result.get("metrics_df")

    return {
        "filename": file.filename,
        "row_count": int(len(df)),
        "run_count": int(df["run_id"].nunique()),
        "has_labels": bool(has_labels),
        "threshold": DETECTION_THRESHOLD,
        "model_weights": MODEL_WEIGHTS,
        "summary": {
            "malicious_runs": int(
                (run_df["verdict"] == "MALICIOUS").sum()
            ),
            "benign_runs": int(
                (run_df["verdict"] == "BENIGN").sum()
            ),
            "mean_final_probability": float(
                run_df["p_final"].mean()
            ),
        },
        "run_results": json.loads(
            run_df.to_json(orient="records")
        ),
        "window_results": (
            json.loads(window_df.to_json(orient="records"))
            if window_df is not None else []
        ),
        "metrics": (
            json.loads(metrics_df.to_json(orient="index"))
            if metrics_df is not None else None
        ),
        "notice": (
            "Predictions use the existing pretrained artifact. "
            "Metrics describe the uploaded labelled dataset and are "
            "not independent validation results."
        ),
    }


@app.get("/api/telemetry/latest")
async def get_latest_telemetry():
    return {
        "telemetry": latest_telemetry,
        "android_window": latest_android_window,
    }


@app.get("/api/detections/latest")
async def get_latest_detection():
    return {"detection": latest_detection}


@app.get("/api/detections/history")
async def get_detection_history(limit: int = 100):
    """Return recent inference results with JSON-safe numeric values."""
    if limit < 1 or limit > MAX_DETECTION_HISTORY:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between 1 and {MAX_DETECTION_HISTORY}.",
        )

    def make_json_safe(value):
        if isinstance(value, dict):
            return {str(k): make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_json_safe(v) for v in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None

        # Handle NumPy scalar values, if present.
        if hasattr(value, "item") and callable(value.item):
            try:
                return make_json_safe(value.item())
            except (ValueError, TypeError):
                pass

        return value

    records = list(reversed(detection_history[-limit:]))

    return {
        "records": [make_json_safe(record) for record in records],
        "count": len(detection_history),
        "storage": "in_memory",
        "notice": "History is cleared when the backend process restarts.",
    }


@app.websocket("/ws/live")
async def live_updates(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)

    try:
        await websocket.send_json({
            "event": "connection",
            "message": "Connected to RANSCOPE-M live updates",
            "telemetry": latest_telemetry,
            "detection": latest_detection,
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass
    finally:
        websocket_clients.discard(websocket)
