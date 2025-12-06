import asyncio
import logging
from io import BytesIO
from typing import Any

from google import genai
from google.genai import types
from PIL import Image as PILImage

from core import config

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

GEMINI_CLIENT: genai.Client | None = None
if config.settings.GEMINI_API_KEY:
    GEMINI_CLIENT = genai.Client(api_key=config.settings.GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. Gemini features will not work.")


class GeminiError(Exception):
    pass


IMAGE_GENERATION_MODEL = "gemini-2.5-flash-image-preview"
TEXT_GENERATION_MODEL = "gemini-2.5-flash"

SAFETY_CONFIGURATION = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]
TEXT_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    response_modalities=["text"],
    safety_settings=SAFETY_CONFIGURATION,
)

UploadableFile = types.File | str | bytes | PILImage.Image


def _uploaded_file_to_image(file: UploadableFile) -> PILImage.Image:
    if isinstance(file, PILImage.Image):
        return file
    if isinstance(file, bytes):
        return PILImage.open(BytesIO(file))
    if isinstance(file, str):
        return PILImage.open(file)
    raise NotImplementedError("Unsupported file type")


def _unpack_response(response: types.GenerateContentResponse) -> Any:
    if not response or not response.candidates:
        raise GeminiError("Failed to generate content")
    parts = response.candidates[0].content.parts
    return parts[0].text


def generate_text(
    prompt: str, file: UploadableFile | None = None, model: str = TEXT_GENERATION_MODEL
):
    if not GEMINI_CLIENT:
        return "Gemini not configured."

    contents: list[Any] = [prompt]
    if file:
        contents.insert(0, _uploaded_file_to_image(file))

    return _unpack_response(
        GEMINI_CLIENT.models.generate_content(
            model=model,
            config=TEXT_GENERATION_CONFIG,
            contents=contents,
        )
    )


async def generate_text_async(prompt: str, file: UploadableFile | None = None):
    return await asyncio.to_thread(generate_text, prompt, file)
