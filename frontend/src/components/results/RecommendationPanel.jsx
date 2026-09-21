/** Business recommendation copy from the API. */
export default function RecommendationPanel({ text }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="text-sm font-medium text-slate-500">Business recommendation</h2>
      <p className="mt-2 text-slate-800">{text ?? "—"}</p>
    </section>
  );
}
