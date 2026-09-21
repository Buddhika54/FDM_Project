/**
 * Form state for all model features (FEATURE_FIELDS).
 */
import { useState } from "react";
import { INITIAL_FORM_STATE } from "../constants/featureFields.js";

export function usePredictionForm() {
  const [values, setValues] = useState(INITIAL_FORM_STATE);
  const [fieldErrors, setFieldErrors] = useState({});

  function onChange(event) {
    const { name, type, value, checked } = event.target;
    setValues((prev) => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
    setFieldErrors((prev) => {
      if (!prev[name]) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  }

  function reset() {
    setValues(INITIAL_FORM_STATE);
    setFieldErrors({});
  }

  return { values, onChange, reset, setValues, fieldErrors, setFieldErrors };
}
