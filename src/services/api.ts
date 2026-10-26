
const API_BASE_URL = "https://ranscope-backend-staging.onrender.com";
const WS_URL = "wss://ranscope-backend-staging.onrender.com/ws/live";

export interface BackendStatus {
  backend: string;
  telemetry_received: boolean;
  detection_available: boolean;
  connected_websocket_clients: number;
  model_inference: string;
  model_load_error?: string | null;
  required_feature_count?: number;
  model_weights?: Record<string, number>;
  threshold?: number;
}

export interface AndroidAggregation {
  features: Record<string, number>;
  available: string[];
  unavailable: string[];
  complete: boolean;
  observed_event_count: number;
  invalid_event_count: number;
  unique_observed_paths: number;
}

export interface TelemetryData {
  device_id: string;
  timestamp: string;
  telemetry: Record<string, unknown>;
  aggregation?: AndroidAggregation;
}

export interface DetectionData {
  device_id: string;
  timestamp: string;
  model_outputs: Record<string, number>;
  model_weights: Record<string, number>;
  final_probability: number;
  risk_level: string;
  threshold: number;
  alert?: boolean;
  window?: number;
  inference_status: string;
  active_models?: string[];
  inactive_models?: string[];
}

export interface TelemetryEvent {
  event: "telemetry_update" | "connection" | "detection_update" | "android_window_update";
  telemetry?: TelemetryData | null;
  android_window?: TelemetryData | null;
  detection?: DetectionData | null;
  message?: string;
}

export interface LatestTelemetryResponse {
  telemetry: TelemetryData | null;
  android_window?: TelemetryData | null;
}

export interface LatestDetectionResponse {
  detection: DetectionData | null;
}

export interface DetectionHistoryResponse {
  records: DetectionData[];
  count: number;
  storage: string;
  notice: string;
}

export interface DetectionResponse {
  accepted: boolean;
  telemetry: TelemetryData;
  detection: DetectionData | null;
  message: string;
}

async function fetchJSON<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail ?? data);

    throw new Error(
      detail || `Request failed with status ${response.status}`,
    );
  }

  return data as T;
}

export function getBackendStatus(): Promise<BackendStatus> {
  return fetchJSON<BackendStatus>("/api/status");
}

export function getLatestTelemetry(): Promise<LatestTelemetryResponse> {
  return fetchJSON<LatestTelemetryResponse>("/api/telemetry/latest");
}

export function getLatestDetection(): Promise<LatestDetectionResponse> {
  return fetchJSON<LatestDetectionResponse>("/api/detections/latest");
}

export function getDetectionHistory(
  limit = 100,
): Promise<DetectionHistoryResponse> {
  return fetchJSON<DetectionHistoryResponse>(
    `/api/detections/history?limit=${limit}`,
  );
}

export function connectLiveUpdates(
  onMessage: (event: TelemetryEvent) => void,
  onOpen?: () => void,
  onClose?: () => void,
  onError?: () => void,
): WebSocket {
  const socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    onOpen?.();
  };

  socket.onmessage = (message: MessageEvent<string>) => {
    try {
      const event = JSON.parse(message.data) as TelemetryEvent;
      onMessage(event);
    } catch (error) {
      console.error("Could not parse WebSocket message:", error);
    }
  };

  socket.onclose = () => {
    onClose?.();
  };

  socket.onerror = () => {
    onError?.();
  };

  return socket;
}

export async function runDetection(
  window: Record<string, number>,
  deviceId = "ranscope-manual-test",
): Promise<DetectionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/detections/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      device_id: deviceId,
      timestamp: new Date().toISOString(),
      window,
    }),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail ?? data);

    throw new Error(
      detail || `Detection request failed: ${response.status}`,
    );
  }

  return data as DetectionResponse;
}
