import React, { useState } from "react";
import { Timer, X } from "lucide-react";
import { estimateCalculation } from "../api/client";

function fmtTime(seconds) {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(1)} hr`;
}

function fmtMem(mb) {
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

export default function TimeEstimator({ xyz, basis, charge, spin }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleEstimate = async () => {
    if (!xyz.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await estimateCalculation({ xyz_input: xyz, basis, charge, spin });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button
        className="btn-secondary w-full flex items-center justify-center gap-2 text-sm py-2"
        onClick={handleEstimate}
        disabled={loading || !xyz.trim()}
      >
        <Timer size={14} />
        {loading ? "Estimating…" : "Estimate Time"}
      </button>

      {error && (
        <div className="mt-2 bg-red-950/40 border border-red-800 rounded-xl px-3 py-2 text-red-300 text-xs">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-2 bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-2.5 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Pre-calculation Estimate
            </span>
            <button
              className="text-slate-500 hover:text-slate-300"
              onClick={() => setResult(null)}
            >
              <X size={14} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-800 rounded-lg p-2">
              <div className="text-slate-400 mb-0.5">Basis functions</div>
              <div className="font-mono text-slate-100 font-semibold">{result.n_basis}</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-2">
              <div className="text-slate-400 mb-0.5">Electrons</div>
              <div className="font-mono text-slate-100 font-semibold">{result.n_electrons}</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-2">
              <div className="text-slate-400 mb-0.5">Dominant integrals</div>
              <div className="font-mono text-slate-100 font-semibold">
                {result.n_dominant_integrals.toLocaleString()}
              </div>
            </div>
            <div className="bg-slate-800 rounded-lg p-2">
              <div className="text-slate-400 mb-0.5">
                {result.method === "RI-JK" ? "B tensor" : "ERI tensor"}
              </div>
              <div className="font-mono text-slate-100 font-semibold">{fmtMem(result.memory_mb)}</div>
            </div>
          </div>

          <div className="bg-slate-800 rounded-lg p-2 text-xs">
            <div className="text-slate-400 mb-0.5">
              Estimated time{" "}
              <span className="text-slate-600">({result.method})</span>
            </div>
            <div className="font-mono text-blue-300 font-semibold text-base">
              {fmtTime(result.estimated_seconds_mid)}
            </div>
            <div className="text-slate-500 mt-0.5">
              range: {fmtTime(result.estimated_seconds_low)} – {fmtTime(result.estimated_seconds_high)}
            </div>
          </div>

          {result.warning && (
            <div className="bg-amber-950/40 border border-amber-800 rounded-lg p-2 text-xs text-amber-300">
              {result.warning}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
