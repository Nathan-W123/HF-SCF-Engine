import React, { useState, useCallback } from "react";
import { Atom, Loader2, Play, AlertCircle, Server } from "lucide-react";
import MoleculeInput from "./components/MoleculeInput";
import BasisSetSelector from "./components/BasisSetSelector";
import MolecularViewer from "./components/MolecularViewer";
import ResultsPanel from "./components/ResultsPanel";
import CalculationHistory from "./components/CalculationHistory";
import TimeEstimator from "./components/TimeEstimator";
import { runCalculation } from "./api/client";

const DEFAULT_XYZ = `O  0.000000  0.000000  0.117176
H  0.000000  0.757001 -0.468704
H  0.000000 -0.757001 -0.468704`;

export default function App() {
  // Molecule state
  const [xyz, setXyz] = useState(DEFAULT_XYZ);
  const [charge, setCharge] = useState(0);
  const [spin, setSpin] = useState(0);
  const [moleculeName, setMoleculeName] = useState("Water");

  // Basis state
  const [zetaLevel, setZetaLevel] = useState("D");
  const [calendarPrefix, setCalendarPrefix] = useState(null);
  const [baseFamily, setBaseFamily] = useState("cc-pV");
  const [selectedRecipeId, setSelectedRecipeId] = useState(null);

  const FIXED_BASES = ["sto-3g", "6-31g", "6-31g*", "6-31g**"];
  const buildBasisName = useCallback(() => {
    if (FIXED_BASES.includes(baseFamily)) return baseFamily;
    if (baseFamily === "def2") {
      const map = { D: "def2-svp", T: "def2-tzvp", Q: "def2-qzvp" };
      return map[zetaLevel] || "def2-svp";
    }
    const cal = calendarPrefix ? `${calendarPrefix}-` : "";
    return `${cal}${baseFamily}${zetaLevel}Z`.toLowerCase();
  }, [baseFamily, zetaLevel, calendarPrefix]);

  // Calculation state
  const [result, setResult] = useState(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [calcError, setCalcError] = useState(null);
  const [maxCycles, setMaxCycles] = useState(200);

  const handleCalculate = useCallback(async () => {
    if (!xyz.trim()) return;
    setIsCalculating(true);
    setCalcError(null);

    try {
      const data = await runCalculation({
        xyz_input: xyz,
        basis: buildBasisName(),
        charge,
        spin,
        max_cycles: maxCycles,
        molecule_name: moleculeName || null,
      });
      setResult(data);
    } catch (e) {
      setCalcError(e.message);
    } finally {
      setIsCalculating(false);
    }
  }, [xyz, buildBasisName, charge, spin, maxCycles, moleculeName]);

  const handleLoadHistory = useCallback((calc) => {
    // Load result from history into results panel (metadata only, no live mol objects)
    setResult({
      ...calc,
      orbitals: [],
      atoms: [],
      dipole: null,
      homo_lumo_gap: calc.homo_energy && calc.lumo_energy
        ? calc.lumo_energy - calc.homo_energy : null,
    });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Atom size={22} className="text-blue-400" />
            <h1 className="text-slate-100 font-bold text-base tracking-tight">
              HF-SCF Molecular Calculator
            </h1>
            <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700">
              Hartree-Fock / RHF
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Server size={12} />
            PySCF backend · SQLite persistence
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Left column: inputs */}
          <div className="lg:col-span-1 space-y-4">
            <MoleculeInput
              xyz={xyz} onChange={setXyz}
              charge={charge} onChargeChange={setCharge}
              spin={spin} onSpinChange={setSpin}
              moleculeName={moleculeName} onNameChange={setMoleculeName}
            />

            <BasisSetSelector
              zetaLevel={zetaLevel} onZetaChange={setZetaLevel}
              calendarPrefix={calendarPrefix} onCalendarChange={setCalendarPrefix}
              baseFamily={baseFamily} onFamilyChange={setBaseFamily}
              selectedRecipeId={selectedRecipeId} onRecipeSelect={setSelectedRecipeId}
            />

            <TimeEstimator
              xyz={xyz}
              basis={buildBasisName()}
              charge={charge}
              spin={spin}
            />

            {/* Advanced SCF options */}
            <div className="card">
              <details>
                <summary className="text-xs text-slate-400 font-semibold uppercase tracking-wider cursor-pointer
                                    hover:text-slate-300 transition-colors">
                  Advanced SCF Options
                </summary>
                <div className="mt-3">
                  <label className="label">Max SCF Cycles</label>
                  <input
                    type="number" min={10} max={1000}
                    className="input-field"
                    value={maxCycles}
                    onChange={(e) => setMaxCycles(parseInt(e.target.value, 10) || 200)}
                  />
                </div>
              </details>
            </div>

            {/* Calculate button */}
            <button
              className="btn-primary w-full py-3 flex items-center justify-center gap-2 text-base"
              onClick={handleCalculate}
              disabled={isCalculating || !xyz.trim()}
            >
              {isCalculating ? (
                <><Loader2 size={18} className="animate-spin" /> Running SCF...</>
              ) : (
                <><Play size={16} /> Run Calculation</>
              )}
            </button>

            {calcError && (
              <div className="bg-red-950/40 border border-red-800 rounded-xl px-4 py-3 flex gap-2">
                <AlertCircle size={16} className="text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-300 text-sm">{calcError}</p>
              </div>
            )}

            <CalculationHistory onLoad={handleLoadHistory} />
          </div>

          {/* Right columns: viewer + results */}
          <div className="lg:col-span-2 space-y-4">
            <MolecularViewer
              atoms={result?.atoms}
              calcId={result?.id}
              orbitals={result?.orbitals}
              homoIdx={result?.homo_idx ?? 0}
            />
            <ResultsPanel result={result} />
          </div>
        </div>
      </main>
    </div>
  );
}
