"""
API endpoint tests using the FastAPI test client.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

H2O_XYZ = """O  0.000000  0.000000  0.117176
H  0.000000  0.757001 -0.468704
H  0.000000 -0.757001 -0.468704"""


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "pyscf_version" in data


class TestBasisOptions:
    def test_basis_options(self, client):
        r = client.get("/basis-options")
        assert r.status_code == 200
        data = r.json()
        assert "zeta_levels" in data
        assert "calendar_prefixes" in data
        assert "base_families" in data
        assert len(data["zeta_levels"]) == 4  # D, T, Q, 5


class TestBasisRecipeCRUD:
    def test_create_recipe(self, client):
        r = client.post("/basis-recipes", json={
            "name": "My DZ Test",
            "zeta_level": "D",
            "base_family": "cc-pV",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "My DZ Test"
        assert data["id"] > 0

    def test_list_recipes(self, client):
        client.post("/basis-recipes", json={"name": "Recipe A", "zeta_level": "D", "base_family": "cc-pV"})
        r = client.get("/basis-recipes")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_duplicate_name_rejected(self, client):
        client.post("/basis-recipes", json={"name": "Unique Name", "zeta_level": "D", "base_family": "cc-pV"})
        r = client.post("/basis-recipes", json={"name": "Unique Name", "zeta_level": "T", "base_family": "cc-pV"})
        assert r.status_code == 409

    def test_delete_recipe(self, client):
        r = client.post("/basis-recipes", json={"name": "To Delete", "zeta_level": "D", "base_family": "cc-pV"})
        recipe_id = r.json()["id"]
        del_r = client.delete(f"/basis-recipes/{recipe_id}")
        assert del_r.status_code == 200
        get_r = client.get(f"/basis-recipes/{recipe_id}")
        assert get_r.status_code == 404

    def test_invalid_calendar_rejected(self, client):
        r = client.post("/basis-recipes", json={
            "name": "Bad Calendar",
            "zeta_level": "D",
            "calendar_prefix": "feb",  # invalid
            "base_family": "cc-pV",
        })
        assert r.status_code == 422


class TestCalculateEndpoint:
    def test_h2o_ccpvdz(self, client):
        r = client.post("/calculate", json={
            "xyz_input": H2O_XYZ,
            "zeta_level": "D",
            "base_family": "cc-pV",
            "molecule_name": "Water",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["converged"] is True
        assert abs(data["total_energy"] - (-76.0268)) < 0.001
        assert data["n_basis"] == 24
        assert data["n_electrons"] == 10
        assert data["id"] > 0

    def test_invalid_xyz_returns_422(self, client):
        r = client.post("/calculate", json={
            "xyz_input": "this is not xyz",
            "zeta_level": "D",
            "base_family": "cc-pV",
        })
        assert r.status_code == 422

    def test_results_stored_in_history(self, client):
        client.post("/calculate", json={
            "xyz_input": H2O_XYZ,
            "zeta_level": "D",
            "base_family": "cc-pV",
            "molecule_name": "Water-History-Test",
        })
        r = client.get("/calculations")
        assert r.status_code == 200
        names = [c["molecule_name"] for c in r.json()]
        assert "Water-History-Test" in names

    def test_delete_calculation(self, client):
        r = client.post("/calculate", json={
            "xyz_input": H2O_XYZ,
            "zeta_level": "D",
            "base_family": "cc-pV",
        })
        calc_id = r.json()["id"]
        del_r = client.delete(f"/calculations/{calc_id}")
        assert del_r.status_code == 200
        get_r = client.get(f"/calculations/{calc_id}")
        assert get_r.status_code == 404
