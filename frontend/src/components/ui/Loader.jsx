export default function Loader({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-600" role="status">
      <span
        className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-brand-700"
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  );
}
