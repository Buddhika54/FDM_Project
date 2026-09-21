/**
 * Home page: problem statement and CTA into the predict form.
 */
import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:py-12">
      <p className="text-sm font-medium uppercase tracking-wide text-brand-700">
        Data Mining Mini Project
      </p>
      <h1 className="mt-2 text-3xl font-bold sm:text-4xl">
        E-Commerce Product Return Risk Prediction
      </h1>
      <p className="mt-4 max-w-3xl text-slate-600">
        Score an order before fulfillment. The dashboard sends the 14 canonical
        predictors to the Flask API, which applies the frozen Phase 7 model and
        recall-first threshold.
      </p>
      <Link
        to="/predict"
        className="mt-8 inline-block rounded-lg bg-brand-700 px-5 py-2.5 text-white hover:bg-brand-900"
      >
        Open prediction form
      </Link>

      <ul className="mt-10 grid gap-4 sm:grid-cols-3">
        <li className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Before fulfillment</h2>
          <p className="mt-2 text-sm text-slate-600">
            Use only features known at order time. Identifiers and the target
            label are never sent.
          </p>
        </li>
        <li className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Same contract as training</h2>
          <p className="mt-2 text-sm text-slate-600">
            Field names, types, and category lists match prediction_schema.py
            and the API.
          </p>
        </li>
        <li className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="font-semibold">Fail loudly</h2>
          <p className="mt-2 text-sm text-slate-600">
            Invalid ratings or categories are blocked in the form, then again
            by the API with field-level errors.
          </p>
        </li>
      </ul>
    </main>
  );
}
