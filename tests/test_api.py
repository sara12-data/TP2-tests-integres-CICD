import pytest
from httpx import AsyncClient
from app.main import app

# Tester une prédiction correcte avec [1.0, 2.0, 3.0]
@pytest.mark.anyio
async def test_predict_correcte():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={
                "features": [1.0, 2.0, 3.0]
            }
        )

    assert resp.status_code == 200
    assert resp.json() == {
        "predictions": [3.0, 6.0, 9.0]
    }

# Tester une prédiction incorrecte avec [1.0, 2.0, 3.0]
@pytest.mark.anyio
async def test_predict_incorrecte():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={
                "features": [1.0, 2.0, 3.0]
            }
        )

    mauvaise_prediction = {
        "predictions": [10.0, 20.0, 30.0]
    }

    assert resp.status_code == 200
    assert resp.json() != mauvaise_prediction

# Tester avec un JSON incorrect
@pytest.mark.anyio
async def test_invalid_json():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={
                "feature1": 3.5,
                "feature2": 1.2,
                "feature3": 4.9
            }
        )

    assert resp.status_code == 422