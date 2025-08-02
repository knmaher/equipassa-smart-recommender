# Equipassa Smart Recommender

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entwicklung
- `src/recommender.py`: Logik
- `src/api/`: FastAPI-Server

## Endpoints
- `GET /recommend/{user_id}?k=5`
