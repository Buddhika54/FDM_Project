/**
 * Results page: last successful prediction (location state or session).
 */
import { Link, useLocation } from "react-router-dom";
import ResultCard from "../components/ResultCard.jsx";
import { readLastPrediction } from "../hooks/usePrediction.js";

export default function ResultsPage() {
  const { state } = useLocation();
  const result = state?.result ?? readLastPrediction();

  if (!result) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-2xl font-bold">No result yet</h1>
        <p className="mt-2 text-slate-600">Submit the prediction form first.</p>
        <Link to="/predict" className="mt-4 inline-block text-brand-700 underline">
          Go to form
        </Link>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 px-4 py-10">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-2xl font-bold">Prediction result</h1>
        <Link to="/predict" className="text-sm text-brand-700 underline">
          Score another order
        </Link>
      </div>
      <ResultCard result={result} />
    </main>
  );
}
