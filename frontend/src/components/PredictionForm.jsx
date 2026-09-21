/**
 * Complete feature form (all model inputs).
 * Client-side required + range checks run before any network call.
 */
import FormField from "./forms/FormField.jsx";
import Button from "./ui/Button.jsx";
import {
  FEATURE_FIELDS,
  validateFormValues,
  valuesToPayload,
} from "../constants/featureFields.js";
import { usePredictionForm } from "../hooks/usePredictionForm.js";

export default function PredictionForm({ onSubmit, disabled = false }) {
  const { values, onChange, fieldErrors, setFieldErrors } = usePredictionForm();

  function handleSubmit(event) {
    event.preventDefault();
    if (disabled) return;

    const nextErrors = validateFormValues(values);
    setFieldErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      return;
    }

    onSubmit(valuesToPayload(values));
  }

  return (
    <form
      noValidate
      onSubmit={handleSubmit}
      className="mt-6 grid gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-2 sm:p-6"
    >
      {FEATURE_FIELDS.map((field) => (
        <FormField
          key={field.name}
          field={field}
          value={values[field.name]}
          onChange={onChange}
          error={fieldErrors[field.name]}
          disabled={disabled}
        />
      ))}
      <div className="sm:col-span-2">
        <Button type="submit" disabled={disabled} className="w-full sm:w-auto">
          Predict return risk
        </Button>
      </div>
    </form>
  );
}
