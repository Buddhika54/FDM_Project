/**
 * Global navbar — links: Home, Predict, Results.
 */
import { NavLink } from "react-router-dom";

const linkClass = ({ isActive }) =>
  `rounded-md px-3 py-2 text-sm font-medium ${
    isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:text-slate-900"
  }`;

export default function Navbar() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <nav className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 py-3">
        <NavLink to="/" className="font-semibold text-slate-900">
          Return Risk
        </NavLink>
        <div className="flex flex-wrap gap-1">
          <NavLink to="/" className={linkClass} end>
            Home
          </NavLink>
          <NavLink to="/predict" className={linkClass}>
            Predict
          </NavLink>
          <NavLink to="/results" className={linkClass}>
            Results
          </NavLink>
        </div>
      </nav>
    </header>
  );
}
