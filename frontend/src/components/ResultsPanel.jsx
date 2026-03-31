import React, { useState } from "react";
import { Zap, Activity, Atom, ChevronDown, ChevronUp, AlertTriangle, CheckCircle2 } from "lucide-react";

const HA_TO_EV = 27.211386;
const HA_TO_KCAL = 627.5094740631;

function EnergyCard({ energy, label }) {
  return (
    <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="energy-value">{energy !== null && energy !== undefined ? energy.toFixed(8) : "—"}</p>
      <p className="text-xs text-slate-500 mt-0.5">Hartree</p>
    </div>
  );
}

function StatRow({ label, value, unit, highlight }) {
  return (
    <div className={`flex items-center justify-between py-1.5 border-b border-slate-800
                     ${highlight ? "text-blue-300" : "text-slate-300"}`}>
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-xs font-mono ${highlight ? "text-blue-300 font-bold" : ""}`}>
        {value ?? "—"}{unit ? <span className="text-slate-500 ml-1">{unit}</span> : null}
      </span>
    </div>
  );
}

export default function ResultsPanel({ result }) {
  const [showAllOrbitals, setShowAllOrbitals] = useState(false);

  if (!result) {
    return (
      <div className="card flex items-center justify-center h-32 text-slate-600 text-sm italic">
        No calculation results yet
      </div>
    );
  }

  const { converged, total_energy, homo_energy, lumo_energy, homo_lumo_gap,
          dipole, orbitals, basis_label, n_electrons, n_basis, error } = result;

  const homoEv = homo_energy ? homo_energy * HA_TO_EV : null;
  const lumoEv = lumo_energy ? lumo_energy * HA_TO_EV : null;
  const gapEv  = homo_lumo_gap ? homo_lumo_gap * HA_TO_EV : null;
  const ipEv   = homo_energy ? -homo_energy * HA_TO_EV : null; // Koopmans' theorem

  const visibleOrbitals = showAllOrbitals ? orbitals : orbitals?.slice(0, 15);

  return (
    <div className="space-y-3">
      {/* Convergence banner */}
      <div className={`card py-3 flex items-center gap-3 ${converged ? "" : "border-red-800"}`}>
        {converged ? (
          <CheckCircle2 size={20} className="text-green-400 flex-shrink-0" />
        ) : (
          <AlertTriangle size={20} className="text-yellow-400 flex-shrink-0" />
        )}
        <div>
          <span className={converged ? "badge-converged" : "badge-diverged"}>
            {converged ? "SCF Converged" : "SCF Did Not Converge"}
          </span>
          <p className="text-xs text-slate-400 mt-0.5">
            {basis_label} · {n_electrons ?? "?"} electrons · {n_basis ?? "?"} basis functions
          </p>
        </div>
        {error && <p className="text-xs text-yellow-400 ml-auto">{error}</p>}
      </div>

      {/* Total energy */}
      <EnergyCard energy={total_energy} label="Total RHF Energy" />

      {/* HOMO/LUMO */}
      <div className="card space-y-1">
        <h3 className="flex items-center gap-1.5 text-slate-300 font-semibold text-xs uppercase tracking-wider mb-2">
          <Zap size={12} className="text-yellow-400" />
          Frontier Orbitals
        </h3>
        <StatRow label="HOMO energy" value={homo_energy?.toFixed(6)} unit="Ha" />
        <StatRow label="HOMO energy" value={homoEv?.toFixed(4)} unit="eV" />
        <StatRow label="LUMO energy" value={lumo_energy?.toFixed(6)} unit="Ha" />
        <StatRow label="LUMO energy" value={lumoEv?.toFixed(4)} unit="eV" />
        <StatRow label="HOMO-LUMO gap" value={gapEv?.toFixed(4)} unit="eV" highlight />
        <StatRow label="Ionization Potential (Koopmans)" value={ipEv?.toFixed(4)} unit="eV" />
      </div>

      {/* Dipole moment */}
      {dipole && (
        <div className="card space-y-1">
          <h3 className="flex items-center gap-1.5 text-slate-300 font-semibold text-xs uppercase tracking-wider mb-2">
            <Activity size={12} className="text-blue-400" />
            Dipole Moment
          </h3>
          <StatRow label="μx" value={dipole.x.toFixed(4)} unit="D" />
          <StatRow label="μy" value={dipole.y.toFixed(4)} unit="D" />
          <StatRow label="μz" value={dipole.z.toFixed(4)} unit="D" />
          <StatRow label="|μ| total" value={dipole.total.toFixed(4)} unit="D" highlight />
        </div>
      )}

      {/* Orbital table */}
      {orbitals && orbitals.length > 0 && (
        <div className="card">
          <h3 className="flex items-center gap-1.5 text-slate-300 font-semibold text-xs uppercase tracking-wider mb-3">
            <Atom size={12} className="text-purple-400" />
            Orbital Energies
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-1.5 text-slate-400 font-semibold">Label</th>
                  <th className="text-right py-1.5 text-slate-400 font-semibold">Occ.</th>
                  <th className="text-right py-1.5 text-slate-400 font-semibold">Energy (Ha)</th>
                  <th className="text-right py-1.5 text-slate-400 font-semibold">Energy (eV)</th>
                </tr>
              </thead>
              <tbody>
                {visibleOrbitals.map((orb) => (
                  <tr
                    key={orb.index}
                    className={`orbital-row border-b border-slate-800/50
                      ${orb.label === "HOMO" ? "homo" : ""}
                      ${orb.label === "LUMO" ? "lumo" : ""}`}
                  >
                    <td className={`py-1.5 font-mono font-semibold
                      ${orb.label === "HOMO" ? "text-blue-300" :
                        orb.label === "LUMO" ? "text-purple-300" :
                        orb.occupation > 0 ? "text-slate-300" : "text-slate-500"}`}>
                      {orb.label}
                    </td>
                    <td className="py-1.5 text-right text-slate-400">{orb.occupation.toFixed(0)}</td>
                    <td className="py-1.5 text-right font-mono text-slate-300">
                      {orb.energy_hartree.toFixed(6)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-slate-400">
                      {orb.energy_ev.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {orbitals.length > 15 && (
            <button
              className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
              onClick={() => setShowAllOrbitals((v) => !v)}
            >
              {showAllOrbitals ? <><ChevronUp size={12} /> Show fewer</> : <><ChevronDown size={12} /> Show all {orbitals.length} orbitals</>}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
