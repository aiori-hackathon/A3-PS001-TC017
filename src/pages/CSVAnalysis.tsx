
import { useState } from "react";
import {
  FileUp,
  LoaderCircle,
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";

const API = "https://ranscope-backend-staging.onrender.com";

type ResultRow = Record<string, string | number>;

type AnalysisResult = {
  filename: string;
  row_count: number;
  run_count: number;
  has_labels: boolean;
  threshold: number;
  model_weights: Record<string, number>;
  summary: {
    malicious_runs: number;
    benign_runs: number;
    mean_final_probability: number;
  };
  run_results: ResultRow[];
  window_results: ResultRow[];
  metrics: Record<string, Record<string, number>> | null;
  notice: string;
};

const pct = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "—";

const panel = "rounded-2xl border border-white/10 bg-[#0d121a] p-5";

export default function CSVAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [showWindows, setShowWindows] = useState(false);

  async function analyze() {
    if (!file) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/api/csv-analysis`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || JSON.stringify(detail),
        );
      }

      setResult(data as AnalysisResult);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Upload failed. Check that the backend is running.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">
          Offline inference
        </p>
        <h1 className="mt-2 text-3xl font-semibold">CSV Analysis</h1>
        <p className="mt-2 text-sm text-white/50">
          Run the existing four pretrained RANSCOPE-M models against unified
          telemetry. Live monitoring remains separate.
        </p>
      </header>

      <section className={`${panel} space-y-4`}>
        <div className="flex items-center gap-3">
          <FileUp className="text-cyan-300" />
          <div>
            <h2 className="font-semibold">Upload telemetry CSV</h2>
            <p className="text-xs text-white/45">
              CSV only · maximum 15 MB · 10,000 rows
            </p>
          </div>
        </div>

        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            setError("");
            setResult(null);
          }}
          className="block w-full text-sm text-white/70 file:mr-3 file:rounded-lg file:border-0 file:bg-white file:px-4 file:py-2 file:font-semibold file:text-black"
        />

        <p className="text-xs leading-5 text-white/40">
          Required columns include run_id, window_start_ms, window_end_ms,
          and all feature columns required by MPVG, SFIM, EAP and CAIT.
          Optional labels: benign/malicious or 0/1.
        </p>

        <button
          onClick={analyze}
          disabled={!file || busy}
          className="rounded-xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 disabled:opacity-50"
        >
          {busy ? (
            <>
              <LoaderCircle
                className="mr-2 inline animate-spin"
                size={16}
              />
              Running models…
            </>
          ) : (
            "Analyze CSV"
          )}
        </button>

        {error && (
          <p
            role="alert"
            className="flex items-start gap-2 text-sm text-rose-200"
          >
            <AlertTriangle size={18} />
            {error}
          </p>
        )}
      </section>

      {result && (
        <>
          <p className="text-sm text-white/50">
            {result.filename} · {result.row_count} windows ·{" "}
            {result.run_count} runs · threshold {pct(result.threshold)}
          </p>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              ["Flagged runs", result.summary.malicious_runs],
              ["Benign verdicts", result.summary.benign_runs],
              ["Mean final score", pct(result.summary.mean_final_probability)],
              ["Ground truth", result.has_labels ? "Available" : "Not supplied"],
            ].map(([heading, value]) => (
              <div className={panel} key={String(heading)}>
                <p className="text-xs uppercase text-white/45">{heading}</p>
                <p className="mt-2 text-2xl font-semibold">{value}</p>
              </div>
            ))}
          </div>

          <section className={panel}>
            <h2 className="font-semibold">Configured model weights</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-4">
              {Object.entries(result.model_weights).map(([name, weight]) => (
                <div className="rounded-xl bg-black/20 p-3" key={name}>
                  <b>{name}</b>
                  <p className="mt-1 text-sm text-cyan-200">{pct(weight)}</p>
                </div>
              ))}
            </div>
          </section>

          {result.metrics && (
            <section className={panel}>
              <h2 className="font-semibold">Metrics on uploaded labels</h2>
              <p className="mt-1 text-xs text-amber-200/70">
                These metrics describe the uploaded dataset; they are not
                independent validation results.
              </p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      {[
                        "Model",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "FPR",
                        "AUC",
                      ].map((heading) => (
                        <th
                          className="p-2 text-left text-white/50"
                          key={heading}
                        >
                          {heading}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.metrics).map(([name, metrics]) => (
                      <tr className="border-t border-white/10" key={name}>
                        <td className="p-2">{name}</td>
                        {[
                          "Accuracy",
                          "Precision",
                          "Recall",
                          "F1",
                          "FPR",
                          "AUC",
                        ].map((key) => (
                          <td className="p-2" key={key}>
                            {pct(metrics[key])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section className={panel}>
            <h2 className="font-semibold">
              Per-run predictions ({result.run_results.length})
            </h2>
            <div className="mt-3 max-h-[520px] overflow-auto">
              <table className="w-full min-w-[800px] text-sm">
                <thead className="sticky top-0 bg-[#0d121a]">
                  <tr>
                    {[
                      "Run ID",
                      "Windows",
                      "MPVG",
                      "SFIM",
                      "EAP",
                      "CAIT",
                      "Final score",
                      "Verdict",
                    ].map((heading) => (
                      <th
                        className="p-2 text-left text-xs text-white/45"
                        key={heading}
                      >
                        {heading}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.run_results.slice(0, 500).map((row, index) => (
                    <tr
                      className="border-t border-white/[.06]"
                      key={`${row.run_id}-${index}`}
                    >
                      <td className="p-2">{row.run_id}</td>
                      <td className="p-2">{row.windows}</td>
                      {["p_MPVG", "p_SFIM", "p_EAP", "p_CAIT", "p_final"].map(
                        (key) => (
                          <td className="p-2" key={key}>
                            {pct(row[key])}
                          </td>
                        ),
                      )}
                      <td className="p-2">
                        {row.verdict === "MALICIOUS" ? (
                          <span className="inline-flex items-center gap-1 text-rose-200">
                            <ShieldAlert size={14} />
                            MALICIOUS
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-emerald-200">
                            <ShieldCheck size={14} />
                            BENIGN
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.run_results.length > 500 && (
              <p className="mt-2 text-xs text-white/40">
                Showing the first 500 runs.
              </p>
            )}
          </section>

          <section className={panel}>
            <button
              onClick={() => setShowWindows((current) => !current)}
              className="flex w-full justify-between font-semibold"
            >
              Per-window results ({result.window_results.length})
              <span className="text-cyan-200">
                {showWindows ? "Hide" : "Show"}
              </span>
            </button>

            {showWindows && (
              <div className="mt-3 max-h-80 overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      {["Run", "Window", "Time (s)", "Final", "Flag"].map(
                        (heading) => (
                          <th className="p-2 text-left" key={heading}>
                            {heading}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {result.window_results.slice(0, 1000).map((row, index) => (
                      <tr
                        key={index}
                        className="border-t border-white/10"
                      >
                        <td className="p-2">{row.run_id}</td>
                        <td className="p-2">{row.window}</td>
                        <td className="p-2">{row.time_s}</td>
                        <td className="p-2">{pct(row.p_final)}</td>
                        <td className="p-2">
                          {row.flag ? "Flagged" : "Clear"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <p className="text-xs leading-5 text-white/40">{result.notice}</p>
        </>
      )}
    </div>
  );
}
