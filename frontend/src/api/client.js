import axios from "axios";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 min — SCF can be slow for large molecules
  headers: { "Content-Type": "application/json" },
});

// Attach request/response interceptors for error normalization
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      "Unknown error";
    return Promise.reject(new Error(message));
  }
);

export const runCalculation = (payload) =>
  api.post("/calculate", payload).then((r) => r.data);

export const getOrbitalCube = (calcId, orbitalIdx, nx = 80, ny = 80, nz = 80) =>
  api
    .post(
      `/calculate/${calcId}/orbital`,
      { calculation_id: calcId, orbital_idx: orbitalIdx, nx, ny, nz },
      { responseType: "text" }
    )
    .then((r) => r.data);

export const listCalculations = (skip = 0, limit = 50) =>
  api.get("/calculations", { params: { skip, limit } }).then((r) => r.data);

export const getCalculation = (id) =>
  api.get(`/calculations/${id}`).then((r) => r.data);

export const deleteCalculation = (id) =>
  api.delete(`/calculations/${id}`).then((r) => r.data);

export const listBasisRecipes = () =>
  api.get("/basis-recipes").then((r) => r.data);

export const createBasisRecipe = (payload) =>
  api.post("/basis-recipes", payload).then((r) => r.data);

export const updateBasisRecipe = (id, payload) =>
  api.put(`/basis-recipes/${id}`, payload).then((r) => r.data);

export const deleteBasisRecipe = (id) =>
  api.delete(`/basis-recipes/${id}`).then((r) => r.data);

export const getBasisOptions = () =>
  api.get("/basis-options").then((r) => r.data);

export const estimateCalculation = (payload) =>
  api.post("/estimate", payload).then((r) => r.data);

export const healthCheck = () =>
  api.get("/health").then((r) => r.data);
