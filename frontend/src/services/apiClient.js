/**
 * Axios instance.
 * Responsibilities: base URL from VITE_API_BASE_URL, JSON headers, shared interceptors later.
 */
import axios from "axios";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000",
  headers: { "Content-Type": "application/json" },
  timeout: 15000,
});

export default apiClient;
