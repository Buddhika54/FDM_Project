/**
 * Predict page: health check + form + result.
 * Axios stays in the service layer; this page only uses hooks.
 */
import { useCallback, useState } from "react";
import HealthBanner from "../components/HealthBanner.jsx";
import PredictionForm from "../components/PredictionForm.jsx";
import ResultCard from "../components/ResultCard.jsx";
import Loader from "../components/ui/Loader.jsx";
import { usePrediction } from "../hooks/usePrediction.js";

function ApiError({ error }) {
  if (!error) return null;
  const details = error.details || {};
  const fields = Object.entries(details);

  return (
    <div
      className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
      role="alert"
    >
      <p className="font-semibold">
        {error.status === 400 ? "Validation failed" : "Prediction failed"}
      </p>
      <p className="mt-1">{error.message}</p>
      {fields.length > 0 ? (
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {fields.map(([field, reason]) => (
            <li key={field}>
              <span className="font-medium">{field}</span>: {reason}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export default function PredictPage() {
  const { submit, loading, error, result } = usePrediction();
  const [health, setHealth] = useState({ checking: true, available: false });
  const onStatus = useCallback((next) => setHealth(next), []);
  const formDisabled = loading || health.checking || !health.available;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:py-10">
      <h1 className="text-2xl font-bold sm:text-3xl">Return risk prediction</h1>
      <p className="mt-2 max-w-3xl text-slate-600">
        Enter features available before fulfillment. The API uses the Phase 7
        model and threshold — it will not silently clip invalid values.
      </p>

      <div className="mt-6">
        <HealthBanner onStatus={onStatus} />
      </div>

      <PredictionForm onSubmit={submit} disabled={formDisabled} />

      {loading ? (
        <div className="mt-6">
          <Loader label="Scoring order…" />
        </div>
      ) : null}

      {error ? (
        <div className="mt-6">
          <ApiError error={error} />
        </div>
      ) : null}

      {result ? (
        <div className="mt-6">
          <ResultCard result={result} />
        </div>
      ) : null}
    </main>
  );
}
