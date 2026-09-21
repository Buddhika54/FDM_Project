import { useCallback, useEffect, useMemo, useState } from "react";
import { checkHealth } from "../services/predictionService.js";

export function useHealth() {
  const [checking, setChecking] = useState(true);
  const [available, setAvailable] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const refresh = useCallback(async () => {
    setChecking(true);
    setError(null);
    try {
      const health = await checkHealth();
      const ready = Boolean(health?.model_loaded && health?.preprocessor_loaded);
      setData(health);
      setAvailable(ready);
      if (!ready) {
        setError("Backend is up but the model or preprocessor is not loaded.");
      }
    } catch {
      setData(null);
      setAvailable(false);
      setError("Backend unavailable. Start the Flask API, then retry.");
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return useMemo(
    () => ({ checking, available, error, data, refresh }),
    [checking, available, error, data, refresh]
  );
}
