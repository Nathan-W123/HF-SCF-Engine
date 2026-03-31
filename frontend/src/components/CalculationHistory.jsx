import React, { useEffect, useState } from "react";
import { History, Trash2, RefreshCw } from "lucide-react";
import { listCalculations, deleteCalculation } from "../api/client";

export default function CalculationHistory({ onLoad }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchHistory = () => {
    setLoading(true);
    listCalculations()
      .then(setHistory)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    await deleteCalculation(id).catch(() => {});
    setHistory((prev) => prev.filter((r) => r.id !== id));
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h2 className="flex items-center gap-2 text-slate-100 font-semibold text-sm">
          <History size={16} className="text-orange-400" />
          Calculation History
        </h2>
        <button
          className="text-slate-400 hover:text-slate-200"
          onClick={fetchHistory}
          title="Refresh"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {history.length === 0 ? (
        <p className="text-xs text-slate-600 italic py-4 text-center">
          {loading ? "Loading..." : "No calculations yet"}
        </p>
      ) : (
        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
          {history.map((calc) => (
            <div
              key={calc.id}
              className="flex items-center justify-between bg-slate-800/50 hover:bg-slate-800
                         border border-slate-700 hover:border-slate-600 rounded-lg px-3 py-2
                         cursor-pointer transition-colors group"
              onClick={() => onLoad(calc)}
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-semibold ${calc.converged ? "text-green-400" : "text-red-400"}`}>
                    {calc.converged ? "✓" : "✗"}
                  </span>
                  <span className="text-xs text-slate-200 font-medium truncate">
                    {calc.molecule_name || `Calc #${calc.id}`}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs font-mono text-slate-500">{calc.basis_label}</span>
                  {calc.total_energy && (
                    <span className="text-xs font-mono text-green-500">
                      {calc.total_energy.toFixed(5)} Ha
                    </span>
                  )}
                </div>
              </div>
              <button
                className="text-slate-600 hover:text-red-400 ml-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => handleDelete(e, calc.id)}
                title="Delete"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
