/**
 * Submit prediction payload; keep loading / error / result in one place
 * so pages and ResultCard stay presentational.
 */
import { useState } from "react";
import {
  formatPredictError,
  predictReturn,
} from "../services/predictionService.js";

const LAST_RESULT_KEY = "lastPrediction";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function submit(payload) {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await predictReturn(payload);
      setResult(data);
      try {
        sessionStorage.setItem(LAST_RESULT_KEY, JSON.stringify(data));
      } catch {
        /* ignore quota / private mode */
      }
      return data;
    } catch (err) {
      const mapped = formatPredictError(err);
      setError(mapped);
      return null;
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setError(null);
    setResult(null);
  }

  return { submit, loading, error, result, reset };
}

export function readLastPrediction() {
  try {
    const raw = sessionStorage.getItem(LAST_RESULT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
