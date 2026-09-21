/**
 * Backend readiness banner. Calls checkHealth via useHealth on mount.
 */
import { useEffect } from "react";
import { useHealth } from "../hooks/useHealth.js";

export default function HealthBanner({ onStatus }) {
  const health = useHealth();
  const { checking, available, data, error, refresh } = health;

  useEffect(() => {
    onStatus?.(health);
  }, [checking, available, data, error, refresh, onStatus, health]);

  if (checking) {
    return (
      <div
        className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600"
        role="status"
      >
        Checking backend…
      </div>
    );
  }

  if (!available) {
    return (
      <div
        className="flex flex-col gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 sm:flex-row sm:items-center sm:justify-between"
        role="alert"
      >
        <div>
          <p className="font-semibold">Backend unavailable</p>
          <p className="mt-1">
            {error ||
              "The prediction API is not reachable. Start Flask with python backend/app.py, then retry."}
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          className="shrink-0 rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-800 hover:bg-red-100"
        >
          Retry
        </button>
      </div>
    );
  }

  const model = data?.model || "model";
  const threshold = data?.threshold;

  return (
    <div
      className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900"
      role="status"
    >
      <span className="font-semibold">Backend ready.</span> {model}
      {threshold != null ? ` · threshold ${threshold}` : ""} · preprocessor and
      model loaded.
    </div>
  );
}
