import React, { useState } from "react";
import { FlaskConical, ChevronDown } from "lucide-react";

const PRESETS = [
  {
    label: "Water (H₂O)",
    xyz: `O  0.000000  0.000000  0.117176
H  0.000000  0.757001 -0.468704
H  0.000000 -0.757001 -0.468704`,
    name: "Water",
  },
  {
    label: "Ammonia (NH₃)",
    xyz: `N  0.000000  0.000000  0.113500
H  0.000000  0.938700 -0.265100
H  0.812900 -0.469400 -0.265100
H -0.812900 -0.469400 -0.265100`,
    name: "Ammonia",
  },
  {
    label: "Methane (CH₄)",
    xyz: `C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118`,
    name: "Methane",
  },
  {
    label: "Hydrogen (H₂)",
    xyz: `H  0.000000  0.000000  0.370000
H  0.000000  0.000000 -0.370000`,
    name: "Hydrogen",
  },
  {
    label: "Carbon Monoxide (CO)",
    xyz: `C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.128000`,
    name: "CO",
  },
  {
    label: "Formaldehyde (H₂CO)",
    xyz: `C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.208000
H  0.000000  0.935000 -0.540000
H  0.000000 -0.935000 -0.540000`,
    name: "Formaldehyde",
  },
];

export default function MoleculeInput({ xyz, onChange, charge, spin, moleculeName, onChargeChange, onSpinChange, onNameChange }) {
  const [showPresets, setShowPresets] = useState(false);

  const applyPreset = (preset) => {
    onChange(preset.xyz);
    onNameChange(preset.name);
    setShowPresets(false);
  };

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-slate-100 font-semibold text-sm">
          <FlaskConical size={16} className="text-blue-400" />
          Molecule Input
        </h2>
        <div className="relative">
          <button
            className="btn-secondary flex items-center gap-1"
            onClick={() => setShowPresets((v) => !v)}
          >
            Presets <ChevronDown size={14} />
          </button>
          {showPresets && (
            <div className="absolute right-0 top-full mt-1 z-10 bg-slate-800 border border-slate-600
                            rounded-lg shadow-xl min-w-[200px] py-1">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  className="w-full text-left px-3 py-2 text-sm text-slate-200
                             hover:bg-slate-700 transition-colors"
                  onClick={() => applyPreset(p)}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Molecule name */}
      <div>
        <label className="label">Molecule Name (optional)</label>
        <input
          type="text"
          className="input-field"
          placeholder="e.g. Water, H2O"
          value={moleculeName}
          onChange={(e) => onNameChange(e.target.value)}
        />
      </div>

      {/* XYZ input */}
      <div>
        <label className="label">
          XYZ Geometry
          <span className="text-slate-500 normal-case font-normal ml-1">(Ångströms)</span>
        </label>
        <textarea
          className="input-field font-mono text-sm resize-y"
          rows={8}
          placeholder={"O  0.000  0.000  0.117\nH  0.000  0.757 -0.471\nH  0.000 -0.757 -0.471"}
          value={xyz}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
        <p className="text-xs text-slate-500 mt-1">
          Format: <code className="font-mono">Symbol  X  Y  Z</code> — one atom per line.
          Optional XYZ header (count + comment) is accepted.
        </p>
      </div>

      {/* Charge & Spin */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label">Charge</label>
          <input
            type="number"
            className="input-field"
            min={-10}
            max={10}
            value={charge}
            onChange={(e) => onChargeChange(parseInt(e.target.value, 10) || 0)}
          />
        </div>
        <div>
          <label className="label">
            Spin Multiplicity (2S)
            <span className="text-slate-500 font-normal normal-case ml-1">0=singlet</span>
          </label>
          <input
            type="number"
            className="input-field"
            min={0}
            max={20}
            value={spin}
            onChange={(e) => onSpinChange(parseInt(e.target.value, 10) || 0)}
          />
        </div>
      </div>
    </div>
  );
}
