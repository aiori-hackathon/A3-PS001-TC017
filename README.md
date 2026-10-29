# A3-PS001-TC017 — Android Ransomware-like Activity Early-Warning in a Controlled Lab

## [Prototype Demonstration Video Link](https://youtu.be/QgVFKGefOUc?si=V36lDIdPoOuNaHjM)

## [Live Prototype Link](http://202.54.170.198:8000)


## RANSCOPE is a research project for monitoring Android application behaviour and analysing telemetry for potential ransomware activity. The project combines Android-side monitoring, a Python backend, pretrained detection models, and a live web dashboard.

YouTube Link: https://youtu.be/fLifYiGd4dQ?si=Y_Wn5dAMMvJEcAs-

## Project components

- **Android application:** Android Studio project under `android/RANSCOPE/`.
- **Monitoring:** Frida-based event collection and Android telemetry collection.
- **Backend:** FastAPI service for telemetry ingestion, feature-window validation, model inference, and live updates.
- **Detection models:** Four pretrained layers—MPVG, SFIM, EAP, and CAIT—combined by the existing detection pipeline.
- **Dashboard:** React, TypeScript, Vite, and Recharts.
- **Dataset utilities:** Python scripts under `scripts/`.

## Architecture

1. Android monitoring produces telemetry and monitoring events.
2. The backend receives events and maintains an in-memory aggregation.
3. A complete feature window can be submitted to the inference endpoint.
4. The detection pipeline produces layer probabilities and a combined risk result.
5. The dashboard receives live updates through WebSockets.

**Important:** Individual Android events and Android feature windows are accepted separately from model inference. The backend does not automatically run inference on every Android event. A complete feature window containing every required model feature must be submitted to `/api/detections/run`.

## Repository structure

```text
RANSCOPE/
├── android/RANSCOPE/             # Android Studio project
├── backend/                      # FastAPI API and detection pipeline
│   ├── main.py
│   ├── event_aggregator.py
│   ├── ranscope_m_unified_backend.py
│   └── models/                   # Pretrained model artifacts
├── public/                       # Dashboard static assets
├── src/                          # React dashboard source
├── submission/android-monitoring/ # Android monitoring utilities
├── scripts/                      # Dataset-generation scripts
├── docs/                         # Project documentation
├── data/                         # Dataset documentation; avoid committing private/raw data
└── submission-materials/         # Presentation and demo materials
```

## Prerequisites

- Git
- Node.js and npm
- Python compatible with the backend dependencies
- Android Studio and an Android emulator or compatible test device for Android testing
- Frida tools if running the Frida monitoring workflow

Versions and platform-specific requirements may depend on the installed Android SDK, Python packages, and Frida setup.

## Run the dashboard

From the repository root:

```bash
npm ci
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

To build the dashboard:

```bash
npm run build
```

To run the linter:

```bash
npm run lint
```

## Run the backend

Create and activate a Python virtual environment from the repository root.

### macOS / Linux

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
```

### Windows PowerShell

```powershell
py -3 -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
```

Install the backend dependencies listed in the backend dependency file, if present. Then start the API from the backend directory:

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

If Uvicorn or another dependency is missing, install the project's declared backend requirements before starting the service.

### Backend endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic service response |
| `GET /health` | Health check |
| `GET /api/status` | Backend, model, and connection status |
| `POST /api/telemetry` | Receive general telemetry |
| `POST /api/telemetry/event` | Receive an individual Android monitoring event |
| `POST /api/telemetry/window` | Store an Android feature window |
| `POST /api/detections/run` | Run inference on a complete feature window |
| `GET /api/telemetry/latest` | Retrieve the latest telemetry |
| `GET /api/detections/latest` | Retrieve the latest detection result |
| `WS /ws/live` | Live dashboard updates |

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The backend currently keeps its latest telemetry, detection, and Android aggregation state in memory. Restarting it clears that runtime state.

## Android and Frida monitoring

1. Open `android/RANSCOPE/` in Android Studio.
2. Allow Android Studio to sync the Gradle project and install any required Android SDK components.
3. Run the application on a compatible emulator or test device.
4. Start the backend and ensure the Android telemetry sender is configured to reach it.
5. For Frida-based collection, follow the setup instructions in `submission/android-monitoring/` and use a compatible Frida client/server pair.

Android emulators may need a different backend address from a physical device. For the standard Android emulator, the host machine is commonly reachable at `10.0.2.2`; for a physical device, use the host's reachable LAN address and ensure the firewall permits the connection.

Only perform monitoring and testing on devices and applications you own or are explicitly authorised to assess.

## Dataset utilities

The `scripts/` directory contains utilities for generating unseen telemetry CSVs and building partner datasets. Review each script's input paths and output destinations before running it. Dataset files are not guaranteed to be included in this repository.

## Cross-platform notes

The dashboard can be developed on macOS, Windows, or Linux with Node.js and npm. The backend requires a compatible Python environment and model dependencies. Android testing requires Android Studio or the relevant Android SDK tools. Frida collection has additional device, architecture, and permission requirements.

Use the commands above from the repository root unless a command explicitly changes directories.

