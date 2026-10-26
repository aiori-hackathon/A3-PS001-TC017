import { useCallback, useEffect, useState } from "react";
import { Activity, RefreshCw, ShieldAlert, ShieldCheck, Cpu, Clock3 } from "lucide-react";
import { getLatestDetection, getLatestTelemetry, type DetectionData, type TelemetryData } from "../services/api";

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-white/[0.025] p-4 min-w-0">
      <p className="text-[10px] font-semibold tracking-[0.16em] text-white/40">{label}</p>
      <p className="mt-2 break-words text-lg font-semibold text-white/90">{value}</p>
      {detail && <p className="mt-1 break-words text-xs text-white/40">{detail}</p>}
    </div>
  );
}

export default function DetectionAnalysis() {
  const [detection, setDetection] = useState<DetectionData | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [latest, latestTelemetry] = await Promise.all([getLatestDetection(), getLatestTelemetry()]);
      setDetection(latest.detection);
      setTelemetry(latestTelemetry.android_window ?? latestTelemetry.telemetry ?? null);
      setUpdatedAt(new Date());
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load live backend data");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const payload = telemetry?.telemetry ?? {};
  const scenario = String(payload.scenario ?? payload.scenario_name ?? "—");
  const observedEvents = payload.observed_event_count ?? payload.file_event_count ?? payload.sfim_event_count ?? "—";
  const observedPaths = payload.unique_observed_paths ?? payload.unique_path_count ?? payload.sfim_unique_path_count ?? "—";
  const active = detection?.active_models ?? Object.keys(detection?.model_outputs ?? {});

  return (
    <div className="space-y-6 pb-10">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold tracking-[0.16em] text-white/35"><Activity size={15} /> LIVE MODEL PIPELINE</div>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">Detection Analysis</h1>
          <p className="mt-1.5 text-sm text-white/45">Live Android scenario telemetry and inference results from the running RANSCOPE backend.</p>
        </div>
        <button onClick={() => void refresh()} className="inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-xs text-white/70 hover:bg-white/5"><RefreshCw size={13} /> Refresh</button>
      </div>

      {error && <div className="rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm text-rose-200">Backend error: {error}</div>}

      <section className="rounded-2xl border border-sky-400/20 bg-sky-400/[0.035] p-5">
        <div className="flex items-center gap-2"><Cpu size={18} className="text-sky-300" /><h2 className="font-semibold">Latest live inference</h2></div>
        {detection ? <>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric label="LATEST VERDICT" value={detection.risk_level} detail={detection.alert ? "Alert raised by backend" : "No backend alert on this window"} />
            <Metric label="FUSED RISK SCORE" value={`${(detection.final_probability * 100).toFixed(3)}%`} detail={`Decision threshold ${(detection.threshold * 100).toFixed(0)}%`} />
            <Metric label="ACTIVE MODELS" value={active.join(", ") || "None"} detail={`${detection.inactive_models?.length ?? 0} inactive layer(s)`} />
            <Metric label="INFERENCE STATUS" value={detection.inference_status || "Unknown"} detail={`Device: ${detection.device_id}`} />
          </div>
          <div className="mt-4">
            <h3 className="mb-3 text-sm font-medium text-white/75">Model outputs</h3>
            {Object.keys(detection.model_outputs ?? {}).length ? <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {Object.entries(detection.model_outputs).map(([name, value]) => <Metric key={name} label={`${name.toUpperCase()} OUTPUT`} value={`${(value * 100).toFixed(3)}%`} detail={`Configured weight: ${((detection.model_weights?.[name.toLowerCase()] ?? detection.model_weights?.[name] ?? 0) * 100).toFixed(0)}%`} />)}
            </div> : <p className="text-sm text-white/40">No per-model outputs were returned for the latest window.</p>}
          </div>
          <p className="mt-4 text-xs text-white/35">Inference timestamp: {new Date(detection.timestamp).toLocaleString()} · {updatedAt ? `Dashboard refreshed ${updatedAt.toLocaleTimeString()}` : "Waiting for refresh"}</p>
        </> : <div className="mt-4 rounded-xl border border-white/[0.07] p-5 text-sm text-white/45">No inference is available yet. Start monitoring or run a scenario in the Android emulator.</div>}
      </section>

      <section className="rounded-2xl border border-white/[0.08] bg-[#0c1017] p-5">
        <div className="flex items-center gap-2"><Activity size={18} className="text-sky-300" /><h2 className="font-semibold">Latest Android scenario</h2></div>
        <p className="mt-1 text-xs text-white/40">This panel shows the latest telemetry received by the backend; it does not invent missing fields.</p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="SCENARIO" value={scenario} />
          <Metric label="OBSERVED EVENTS" value={String(observedEvents)} />
          <Metric label="OBSERVED PATHS" value={String(observedPaths)} />
          <Metric label="TELEMETRY EVENT" value={String(payload.event ?? "—")} />
        </div>
      </section>

      <section className="rounded-2xl border border-white/[0.08] bg-[#0c1017] p-5">
        <div className="flex items-center gap-2"><ShieldAlert size={18} className="text-amber-300" /><h2 className="font-semibold">Model availability</h2></div>
        <p className="mt-1 text-xs leading-5 text-white/45">Only models whose required input features are available are included in the live result. Missing measurements are not fabricated.</p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {["MPVG", "SFIM", "EAP", "CAIT"].map((name) => {
            const isActive = active.some((model) => model.toUpperCase() === name);
            return <div key={name} className="flex items-center gap-3 rounded-xl border border-white/[0.08] p-4">
              {isActive ? <ShieldCheck size={18} className="text-emerald-300" /> : <Clock3 size={18} className="text-white/35" />}
              <div><p className="text-sm font-semibold">{name}</p><p className={`mt-1 text-xs ${isActive ? "text-emerald-300" : "text-white/40"}`}>{isActive ? "Active for latest window" : "Waiting for required telemetry"}</p></div>
            </div>;
          })}
        </div>
      </section>
    </div>
  );
}
