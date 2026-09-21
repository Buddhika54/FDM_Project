/**
 * Prediction API service.
 * Pages/components must not call Axios or fetch directly.
 */
import apiClient from "./apiClient.js";

export async function checkHealth() {
  const { data } = await apiClient.get("/api/health");
  return data;
}

export async function predictReturn(payload) {
  const { data } = await apiClient.post("/api/predict", payload);
  return data;
}

/** Map Axios / Flask errors into a UI-safe { message, details, status }. */
export function formatPredictError(err) {
  const status = err.response?.status;
  const data = err.response?.data;

  if (!err.response) {
    return {
      message:
        "Backend unavailable. Start the Flask API at http://127.0.0.1:5000.",
      details: {},
      status: 0,
    };
  }

  const details =
    data?.details && typeof data.details === "object" ? data.details : {};
  const detailList = Object.entries(details)
    .map(([field, reason]) => `${field}: ${reason}`)
    .join("; ");

  let message = data?.message || err.message || "Prediction request failed.";
  if (status === 400 && detailList) {
    message = `${message} ${detailList}`;
  } else if (status === 500) {
    message = data?.message || "Prediction failed.";
  } else if (status === 503) {
    message =
      data?.message ||
      "Model or preprocessor is not loaded. Train and save artifacts to models/.";
  }

  return { message, details, status };
}
