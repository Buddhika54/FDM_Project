/**
 * Reusable labeled control (number, select, or coupon checkbox).
 */
export default function FormField({ field, value, onChange, error, disabled }) {
  const id = field.name;
  const invalid = Boolean(error);
  const controlClass = `rounded-md border px-3 py-2 text-slate-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-500 ${
    invalid ? "border-red-400 focus:ring-red-400" : "border-slate-300"
  } disabled:cursor-not-allowed disabled:bg-slate-100`;

  if (field.type === "checkbox") {
    return (
      <label className="flex items-start gap-3 text-sm sm:col-span-2">
        <input
          id={id}
          name={field.name}
          type="checkbox"
          checked={Boolean(value)}
          onChange={onChange}
          disabled={disabled}
          className="mt-1 h-4 w-4 rounded border-slate-300 text-brand-700 focus:ring-brand-500"
        />
        <span>
          <span className="font-medium text-slate-700">{field.label}</span>
          {field.hint ? (
            <span className="mt-0.5 block text-xs text-slate-500">{field.hint}</span>
          ) : null}
        </span>
      </label>
    );
  }

  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-medium text-slate-700">
        {field.label}
        <span className="text-red-600"> *</span>
      </span>
      {field.type === "select" ? (
        <select
          id={id}
          name={field.name}
          value={value}
          onChange={onChange}
          required
          disabled={disabled}
          aria-invalid={invalid}
          className={controlClass}
        >
          <option value="">Select</option>
          {(field.options || []).map((opt) => (
            <option key={String(opt.value)} value={opt.value}>
              {opt.label ?? opt.value}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          name={field.name}
          type="number"
          step={field.step}
          min={field.min}
          max={field.max}
          value={value}
          onChange={onChange}
          required
          disabled={disabled}
          aria-invalid={invalid}
          inputMode={field.integer ? "numeric" : "decimal"}
          className={controlClass}
        />
      )}
      {field.hint ? <span className="text-xs text-slate-500">{field.hint}</span> : null}
      {error ? <span className="text-xs text-red-600">{error}</span> : null}
    </label>
  );
}
