/**
 * Visual prediction outcome: probability %, flagged banner, model identity.
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  const probability = Number(
    result.return_risk_probability ?? result.probability ?? 0
  );
  const percent =
    result.probability_percent != null
      ? Number(result.probability_percent)
      : Math.round(probability * 10000) / 100;
  const flagged = Boolean(result.flagged);
  const modelName = result.model || result.model_name || "Unknown model";
  const modelVersion = result.model_version || result.version || "—";

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div
        className={`px-4 py-3 text-sm font-semibold sm:px-6 ${
          flagged
            ? "bg-red-600 text-white"
            : "bg-emerald-600 text-white"
        }`}
        role="status"
      >
        {flagged ? "High Return Risk" : "Low Return Risk"}
        {result.threshold != null ? (
          <span className="ml-2 font-normal opacity-90">
            (flagged if probability ≥ {result.threshold})
          </span>
        ) : null}
      </div>

      <div className="space-y-4 p-4 sm:p-6">
        <div>
          <p className="text-sm font-medium text-slate-500">Return risk probability</p>
          <p className="mt-1 text-3xl font-bold tabular-nums text-slate-900">
            {percent.toFixed(2)}%
          </p>
        </div>

        <dl className="grid gap-3 text-sm sm:grid-cols-2">
          <div>
            <dt className="text-slate-500">Flagged</dt>
            <dd className="font-medium text-slate-900">{flagged ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Risk band</dt>
            <dd className="font-medium text-slate-900">{result.risk_level ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Model</dt>
            <dd className="font-medium text-slate-900">{modelName}</dd>
          </div>
          <div>
            <dt className="text-slate-500">Version</dt>
            <dd className="break-all font-medium text-slate-900">{modelVersion}</dd>
          </div>
        </dl>

        {result.recommendation ? (
          <p className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700">
            {result.recommendation}
          </p>
        ) : null}
      </div>
    </section>
  );
}
