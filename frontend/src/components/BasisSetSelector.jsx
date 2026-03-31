import React, { useEffect, useState } from "react";
import { BookOpen, Plus, Trash2, X } from "lucide-react";
import { getBasisOptions, listBasisRecipes, createBasisRecipe, deleteBasisRecipe } from "../api/client";

const DEFAULT_OPTIONS = {
  zeta_levels: [
    { value: "D", label: "Double-ζ (DZ)" },
    { value: "T", label: "Triple-ζ (TZ)" },
    { value: "Q", label: "Quadruple-ζ (QZ)" },
    { value: "5", label: "Quintuple-ζ (5Z)" },
  ],
  calendar_prefixes: [
    { value: null, label: "None (no diffuse)" },
    { value: "aug", label: "aug- (full augmentation)" },
    { value: "jul", label: "jul- (drop highest AM)" },
    { value: "jun", label: "jun- (drop 2 highest AM)" },
    { value: "may", label: "may- (drop 3 highest AM)" },
    { value: "apr", label: "apr- (drop 4 highest AM)" },
    { value: "mar", label: "mar- (s-diffuse only)" },
  ],
  base_families: [
    { value: "cc-pV", label: "Dunning cc-pV (correlation-consistent)" },
    { value: "cc-pwCV", label: "Dunning cc-pwCV (core-valence)" },
    { value: "def2", label: "Weigend-Ahlrichs def2" },
  ],
};

function BasisRecipeModal({ onClose, onSave }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [zetaLevel, setZetaLevel] = useState("D");
  const [calPrefix, setCalPrefix] = useState("");
  const [baseFamily, setBaseFamily] = useState("cc-pV");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const handleSave = async () => {
    if (!name.trim()) { setError("Name is required"); return; }
    setSaving(true);
    setError("");
    try {
      const recipe = await createBasisRecipe({
        name: name.trim(),
        description: description.trim() || null,
        zeta_level: zetaLevel,
        calendar_prefix: calPrefix || null,
        base_family: baseFamily,
      });
      onSave(recipe);
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-md shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-slate-100 font-semibold flex items-center gap-2">
            <BookOpen size={16} className="text-blue-400" />
            Save Basis Recipe
          </h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200">
            <X size={18} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="label">Recipe Name *</label>
            <input
              className="input-field"
              placeholder="My Custom Triple-Zeta + Jul Diffuse"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div>
            <label className="label">Description</label>
            <input
              className="input-field"
              placeholder="Optional description..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="label">Family</label>
              <select className="select-field text-sm" value={baseFamily} onChange={(e) => setBaseFamily(e.target.value)}>
                {DEFAULT_OPTIONS.base_families.map((f) => (
                  <option key={f.value} value={f.value}>{f.value}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Zeta</label>
              <select className="select-field text-sm" value={zetaLevel} onChange={(e) => setZetaLevel(e.target.value)}>
                {DEFAULT_OPTIONS.zeta_levels.map((z) => (
                  <option key={z.value} value={z.value}>{z.value}Z</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Diffuse</label>
              <select className="select-field text-sm" value={calPrefix} onChange={(e) => setCalPrefix(e.target.value)}>
                <option value="">None</option>
                {DEFAULT_OPTIONS.calendar_prefixes.slice(1).map((c) => (
                  <option key={c.value} value={c.value}>{c.value}-</option>
                ))}
              </select>
            </div>
          </div>

          {error && (
            <p className="text-red-400 text-sm bg-red-950/30 border border-red-800 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
        </div>

        <div className="flex gap-2 mt-5 justify-end">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Recipe"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function BasisSetSelector({
  zetaLevel, onZetaChange,
  calendarPrefix, onCalendarChange,
  baseFamily, onFamilyChange,
  selectedRecipeId, onRecipeSelect,
}) {
  const [options] = useState(DEFAULT_OPTIONS);
  const [recipes, setRecipes] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [loadingRecipes, setLoadingRecipes] = useState(true);

  useEffect(() => {
    setLoadingRecipes(true);
    listBasisRecipes()
      .then((data) => setRecipes(Array.isArray(data) ? data : []))
      .catch(() => setRecipes([]))
      .finally(() => setLoadingRecipes(false));
  }, []);

  const handleDeleteRecipe = async (e, id) => {
    e.stopPropagation();
    await deleteBasisRecipe(id).catch(() => {});
    setRecipes((prev) => prev.filter((r) => r.id !== id));
    if (selectedRecipeId === id) onRecipeSelect(null);
  };

  // Compute preview label
  const calLabel = calendarPrefix ? `${calendarPrefix}-` : "";
  const preview = baseFamily === "def2"
    ? `def2-${zetaLevel === "D" ? "SVP" : zetaLevel === "T" ? "TZVP" : "QZVP"}`
    : `${calLabel}${baseFamily}${zetaLevel}Z`;

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-slate-100 font-semibold text-sm">
          <BookOpen size={16} className="text-purple-400" />
          Basis Set
        </h2>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-purple-300 bg-purple-950/40 px-2 py-0.5 rounded border border-purple-800">
            {preview}
          </span>
        </div>
      </div>

      {/* If recipe selected, show override banner */}
      {selectedRecipeId && (
        <div className="bg-blue-950/40 border border-blue-800 rounded-lg px-3 py-2 flex items-center justify-between">
          <span className="text-blue-300 text-xs font-semibold">
            Using saved recipe #{selectedRecipeId}
          </span>
          <button
            className="text-slate-400 hover:text-slate-200 text-xs"
            onClick={() => onRecipeSelect(null)}
          >
            Clear
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3">
        {/* Base Family */}
        <div>
          <label className="label">Basis Family</label>
          <select
            className="select-field"
            value={baseFamily}
            onChange={(e) => onFamilyChange(e.target.value)}
            disabled={!!selectedRecipeId}
          >
            {options.base_families.map((f) => (
              <option key={f.value} value={f.value}>{f.label}</option>
            ))}
          </select>
        </div>

        {/* Zeta Level */}
        <div>
          <label className="label">Zeta Level</label>
          <div className="grid grid-cols-4 gap-1.5">
            {options.zeta_levels.map((z) => (
              <button
                key={z.value}
                className={`py-1.5 rounded-lg text-sm font-semibold border transition-colors
                  ${zetaLevel === z.value && !selectedRecipeId
                    ? "bg-blue-600 border-blue-500 text-white"
                    : "bg-slate-800 border-slate-600 text-slate-300 hover:border-slate-500"
                  } ${selectedRecipeId ? "opacity-50 cursor-not-allowed" : ""}`}
                onClick={() => !selectedRecipeId && onZetaChange(z.value)}
              >
                {z.value}Z
              </button>
            ))}
          </div>
        </div>

        {/* Calendar Convention */}
        <div>
          <label className="label">
            Diffuse Functions
            <span className="text-slate-500 normal-case font-normal ml-1">(Calendar convention)</span>
          </label>
          <div className="grid grid-cols-4 gap-1.5">
            {[{ value: null, short: "None" }, ...options.calendar_prefixes.slice(1).map(c => ({ value: c.value, short: `${c.value}-` }))].map((c) => (
              <button
                key={String(c.value)}
                className={`py-1.5 rounded-lg text-xs font-semibold border transition-colors
                  ${(calendarPrefix ?? null) === c.value && !selectedRecipeId
                    ? "bg-purple-700 border-purple-500 text-white"
                    : "bg-slate-800 border-slate-600 text-slate-300 hover:border-slate-500"
                  } ${selectedRecipeId ? "opacity-50 cursor-not-allowed" : ""}`}
                onClick={() => !selectedRecipeId && onCalendarChange(c.value)}
                title={c.value ? `${c.value}- diffuse functions` : "No diffuse augmentation"}
              >
                {c.short}
              </button>
            ))}
          </div>
          {calendarPrefix && (
            <p className="text-xs text-slate-500 mt-1.5">
              {calendarPrefix === "aug" && "Full augmentation — adds diffuse functions for all angular momenta"}
              {calendarPrefix === "jul" && "Drops highest-AM diffuse shell (Papajak et al.)"}
              {calendarPrefix === "jun" && "Drops two highest-AM diffuse shells"}
              {calendarPrefix === "may" && "Drops three highest-AM diffuse shells (s+p diffuse only)"}
              {calendarPrefix === "apr" && "Drops four highest-AM diffuse shells"}
              {calendarPrefix === "mar" && "s-diffuse only"}
            </p>
          )}
        </div>
      </div>

      {/* Saved Recipes */}
      <div className="border-t border-slate-700 pt-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Saved Recipes
          </span>
          <button
            className="text-blue-400 hover:text-blue-300 text-xs flex items-center gap-1"
            onClick={() => setShowModal(true)}
          >
            <Plus size={12} /> Save current
          </button>
        </div>

        {loadingRecipes ? (
          <p className="text-xs text-slate-500">Loading recipes...</p>
        ) : recipes.length === 0 ? (
          <p className="text-xs text-slate-600 italic">No saved recipes yet</p>
        ) : (
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {recipes.map((r) => (
              <div
                key={r.id}
                className={`flex items-center justify-between px-2 py-1.5 rounded-lg cursor-pointer
                            border text-xs transition-colors
                            ${selectedRecipeId === r.id
                              ? "bg-blue-950/50 border-blue-700 text-blue-200"
                              : "bg-slate-800/50 border-slate-700 text-slate-300 hover:border-slate-600"
                            }`}
                onClick={() => onRecipeSelect(selectedRecipeId === r.id ? null : r.id)}
              >
                <div>
                  <span className="font-semibold">{r.name}</span>
                  {r.pyscf_basis_str && (
                    <span className="ml-2 text-slate-500 font-mono">{r.pyscf_basis_str}</span>
                  )}
                </div>
                <button
                  className="text-slate-500 hover:text-red-400 ml-2"
                  onClick={(e) => handleDeleteRecipe(e, r.id)}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {showModal && (
        <BasisRecipeModal
          onClose={() => setShowModal(false)}
          onSave={(recipe) => setRecipes((prev) => [recipe, ...prev])}
          initialZeta={zetaLevel}
          initialCal={calendarPrefix}
          initialFamily={baseFamily}
        />
      )}
    </div>
  );
}
