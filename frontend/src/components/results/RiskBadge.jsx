/** Risk chip: Low | Medium | High. Color by level in Phase 8. */
export default function RiskBadge({ level }) {
  return (
    <p className="text-sm">
      Risk level:{" "}
      <span className="rounded-full bg-slate-200 px-3 py-1 font-medium">
        {level ?? "—"}
      </span>
    </p>
  );
}
