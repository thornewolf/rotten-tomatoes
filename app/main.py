from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from services.gemini_integration import generate_text_async
from services.kalshi_api import fetch_markets_from_kalshi

app = FastAPI(title="Rotten Tomatoes Predictor")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/")
async def home(request: Request):
    markets = fetch_markets_from_kalshi()
    return templates.TemplateResponse("index.html", {"request": request, "markets": markets})


@app.get("/ask-ai")
async def ask_ai(prompt: str):
    response = await generate_text_async(prompt)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
