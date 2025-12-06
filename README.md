This repo uses `uv` for package management.
The purpose of this project is to predict rotten tomatoes markets on the prediction platform Kalshi.
We believe that we can predict the future rating of a movie based on the ratings that have come in so far.
We believe we can predict the rating at N days into the market given inputs (days since release, ratings so far), though we are open to multiple estimation methods.

The general structure of this project should have:

1. notebooks with some standard ml libraries imported for doing random forest classification
2. scraper scripts that grab kalshi wbe pages
3. processor scripts that convert fetched web pages into tabular data for post-processing
4. model script that trains on the tabular data and reports metrics
5. a kalshi integration to fetch / search markets exposed as functions
6. a gemini integration for general ai stuff as needed
7. a web interface that can be extended to do misc tasks / trigger all this stuff mentioned before

I am gonna attach some relevant code from other projects I have built:
```
  # kalshi_api.py
  
import os
from typing import List, Dict, Any
import logging
from security.kalshi_signer import kalshi_request, is_auth_configured

logger = logging.getLogger(__name__)


def fetch_markets_from_kalshi() -> List[Dict[str, Any]]:
    """
    Fetch markets from Kalshi API using proper request signing.
    Falls back to mock data if API is not available or not configured.
    """
    try:
        if not is_auth_configured():
            logger.warning("Kalshi authentication not configured, using mock data")
            return get_mock_markets()
        
        # Use the authenticated kalshi_request function
        response = kalshi_request(
            "GET", 
            "/trade-api/v2/markets?status=open&limit=50",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            markets = data.get("markets", [])
            logger.info(f"Successfully fetched {len(markets)} markets from Kalshi API")
            
            return [
                {
                    "id": market.get("ticker", f"market_{i}"),
                    "title": market.get("title", f"Market {i}"),
                    "category": market.get("category", "General"),
                    "status": market.get("status", "open"),
                    "yes_price": market.get("yes_bid", 0.5)
                }
                for i, market in enumerate(markets[:20])  # Limit for MVP
            ]
        else:
            logger.warning(f"Kalshi API error: {response.status_code}, using mock data")
            return get_mock_markets()
            
    except Exception as e:
        logger.error(f"Error fetching from Kalshi API: {e}, using mock data")
        return get_mock_markets()


def get_mock_markets() -> List[Dict[str, Any]]:
    """Mock market data for development and testing."""
    return [
        {
            "id": "INF-DEC24",
            "title": "Will US inflation rate exceed 3% in December 2024?",
            "category": "Economics",
            "status": "open",
            "yes_price": 0.65
        },
        {
            "id": "FED-JAN25", 
            "title": "Will the Federal Reserve raise interest rates in January 2025?",
            "category": "Economics",
            "status": "open",
            "yes_price": 0.25
        },
        {
            "id": "TECH-Q1",
            "title": "Will any major tech stock gain more than 20% in Q1 2025?",
            "category": "Technology",
            "status": "open",
            "yes_price": 0.45
        },
        {
            "id": "WEATHER-JAN",
            "title": "Will January 2025 be warmer than average in the US?",
            "category": "Weather",
            "status": "open", 
            "yes_price": 0.55
        },
        {
            "id": "ELECTION-LOCAL",
            "title": "Will turnout exceed 60% in the next major city election?",
            "category": "Politics",
            "status": "open",
            "yes_price": 0.40
        },
        {
            "id": "SPORTS-SUPERBOWL",
            "title": "Will the Super Bowl 2025 total score exceed 50 points?",
            "category": "Sports", 
            "status": "open",
            "yes_price": 0.70
        },
        {
            "id": "CRYPTO-BTC",
            "title": "Will Bitcoin reach $100,000 before June 2025?",
            "category": "Cryptocurrency",
            "status": "open",
            "yes_price": 0.35
        },
        {
            "id": "AI-BREAKTHROUGH",
            "title": "Will a new major AI breakthrough be announced in Q1 2025?",
            "category": "Technology",
            "status": "open",
            "yes_price": 0.60
        }
    ]
```
```
# gemini_integration.py

import asyncio
import logging
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image as PILImage

from core import config

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

if not config.settings.GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY not set. Gemini features will not work.")
GEMINI_CLIENT = genai.Client(api_key=config.settings.GEMINI_API_KEY)


class GeminiError(Exception):
    pass


IMAGE_GENERATION_MODEL = "gemini-2.5-flash-image-preview"
TEXT_GENERATION_MODEL = "gemini-2.5-flash"
# Intentionally disable all safety filters permanently.
SAFETY_CONFIGURATION = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]
IMAGE_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    response_modalities=["image", "text"],
    safety_settings=SAFETY_CONFIGURATION,
)
TEXT_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    response_modalities=["text"],
    safety_settings=SAFETY_CONFIGURATION,
)

type UploadableFile = types.File | str | bytes | PILImage.Image


def _image_input_to_part(client: genai.Client, file: UploadableFile) -> types.Part:
    """Convert various image input types to a Gemini Part."""
    if isinstance(file, bytes):
        return types.Part.from_bytes(data=file, mime_type="image/jpeg")
    if isinstance(file, str):
        uploaded_file = client.files.upload(file=file)
        if uploaded_file.uri is None:
            raise ValueError("Failed to upload file")
        return types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type)
    raise NotImplementedError("Unsupported file type")


def _uploaded_file_to_image(file: UploadableFile) -> PILImage.Image:
    """Convert various image input types to a PIL Image."""
    if isinstance(file, PILImage.Image):
        return file
    if isinstance(file, bytes):
        return PILImage.open(BytesIO(file))
    if isinstance(file, str):
        return PILImage.open(file)
    raise NotImplementedError("Unsupported file type")


def _unpack_response(response: types.GenerateContentResponse) -> str | bytes:
    if (
        not response
        or not response.candidates
        or not response.candidates[0].content
        or not response.candidates[0].content.parts
    ):
        raise GeminiError("Failed to generate content")
    parts = response.candidates[0].content.parts
    has_inline_data = any(part.inline_data for part in parts)
    logger.info(f"FLAG Number of parts: {len(parts)}")
    if has_inline_data:
        image_bytes = next(p.inline_data.data for p in parts if p.inline_data)
        if not image_bytes:
            raise ValueError("Failed to get image from inline data.")
        return image_bytes
    text_response = parts[0].text
    if not text_response:
        raise ValueError("Failed to get text from response.")
    return text_response


def _generate_content(
    client: genai.Client,
    model: str,
    prompt: str,
    file: UploadableFile | None,
):
    contents: list[str | PILImage.Image] = []
    if file:
        contents = [_uploaded_file_to_image(file), prompt]
    else:
        contents = [prompt]

    if model in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-flash-lite-latest"]:
        config = TEXT_GENERATION_CONFIG
    elif model == IMAGE_GENERATION_MODEL:
        config = IMAGE_GENERATION_CONFIG
    else:
        raise ValueError(f"Unsupported model: {model}")
    return _unpack_response(client.models.generate_content(model=model, config=config, contents=contents))


def generate_text(prompt: str, file: UploadableFile | None = None, model: str = TEXT_GENERATION_MODEL):
    return _generate_content(GEMINI_CLIENT, model, prompt, file)


async def generate_text_async(prompt: str, file: UploadableFile | None = None):
    return await asyncio.to_thread(generate_text, prompt, file)


def generate_image(prompt: str, file: UploadableFile | None = None):
    return _generate_content(GEMINI_CLIENT, IMAGE_GENERATION_MODEL, prompt, file)


async def generate_image_async(prompt: str, file: UploadableFile | None = None) -> bytes:
    result = await asyncio.to_thread(generate_image, prompt, file)
    assert isinstance(result, bytes), "Expected bytes from image generation"
    return result
```

`tree app database services scripts models/ core | pbcopy`
```
app
├── __init__.py
├── __pycache__
│   └── main.cpython-313.pyc
├── api
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── auth.cpython-313.pyc
│   │   ├── blog.cpython-313.pyc
│   │   ├── deps.cpython-313.pyc
│   │   ├── exception_handling.cpython-313.pyc
│   │   ├── exceptions.cpython-313.pyc
│   │   ├── feedback.cpython-313.pyc
│   │   ├── images.cpython-313.pyc
│   │   ├── page_routes.cpython-313.pyc
│   │   ├── products.cpython-313.pyc
│   │   ├── subscriptions.cpython-313.pyc
│   │   └── users.cpython-313.pyc
│   ├── auth.py
│   ├── feedback.py
│   ├── images.py
│   ├── products.py
│   ├── subscriptions.py
│   └── users.py
├── deps
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── tokens.cpython-313.pyc
│   │   └── users.cpython-313.pyc
│   ├── tokens.py
│   └── users.py
├── main.py
├── page_routes
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── account_pages.cpython-313.pyc
│   │   ├── app_pages.cpython-313.pyc
│   │   ├── auth_pages.cpython-313.pyc
│   │   ├── feedback_pages.cpython-313.pyc
│   │   └── landing_pages.cpython-313.pyc
│   ├── account_pages.py
│   ├── app_pages.py
│   ├── auth_pages.py
│   ├── feedback_pages.py
│   └── landing_pages.py
├── static
│   ├── css
│   │   ├── styles.css
│   │   └── tailwind.css
│   ├── export
│   ├── images
│   │   ├── amazon.png
│   │   ├── cattoy-gen.png
│   │   ├── etsy.png
│   │   ├── laptop-gen.png
│   │   ├── laptop.png
│   │   ├── link-preview.png
│   │   ├── notebook.png
│   │   ├── pip_cattoy.png
│   │   ├── pip_shoe.png
│   │   ├── pip_stanley.png
│   │   ├── pip_tote.png
│   │   ├── shopify.png
│   │   ├── tote-lifestyle2.png
│   │   ├── tote-lifestyle3.png
│   │   ├── tote.png
│   │   ├── wellness_product_input.jpg
│   │   └── wellness_product_lifestyle.png
│   ├── js
│   │   ├── main.js
│   │   ├── product_detail.js
│   │   ├── products_list.js
│   │   └── settings.js
│   └── optimize_images.py
├── templates
│   ├── account_pages
│   │   ├── purchase_cancel.html
│   │   ├── purchase_success.html
│   │   └── settings.html
│   ├── app_pages
│   │   ├── dashboard.html
│   │   ├── product_detail.html
│   │   └── products_list.html
│   ├── auth_pages
│   │   └── signin.html
│   ├── base.html
│   ├── feedback_pages
│   │   └── feedback.html
│   ├── index.html
│   ├── landing_pages
│   │   ├── dropshipping.html
│   │   └── image_editing.html
│   ├── partials
│   │   ├── add_product_modal.html
│   │   ├── feedback_form.html
│   │   ├── image_card.html
│   │   ├── image_list.html
│   │   ├── login_form.html
│   │   └── product_card.html
│   └── pricing.html
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-313.pyc
    │   ├── blog_utils.cpython-313.pyc
    │   ├── exception_handling.cpython-313.pyc
    │   ├── htmx_snippets.cpython-313.pyc
    │   ├── model_to_response_adapters.cpython-313.pyc
    │   └── username.cpython-313.pyc
    ├── exception_handling.py
    ├── htmx_snippets.py
    ├── model_to_response_adapters.py
    └── username.py
database
├── __pycache__
│   └── db.cpython-313.pyc
└── db.py
services
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   ├── gemini_integration.cpython-313.pyc
│   ├── image_storage_service.cpython-313.pyc
│   ├── lifestyle_image_service.cpython-313.pyc
│   ├── notification_service.cpython-313.pyc
│   ├── ntfy_integration.cpython-313.pyc
│   ├── stripe_integration.cpython-313.pyc
│   ├── user_creation_util.cpython-313.pyc
│   └── user_quota_service.cpython-313.pyc
├── gemini_integration.py
├── lifestyle_image_service.py
├── notification_service.py
├── ntfy_integration.py
├── storage
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── base.cpython-313.pyc
│   │   ├── hybrid.cpython-313.pyc
│   │   ├── local.cpython-313.pyc
│   │   └── s3.cpython-313.pyc
│   ├── base.py
│   ├── hybrid.py
│   ├── local.py
│   └── s3.py
├── stripe_integration.py
├── user_creation_util.py
└── user_quota_service.py
scripts
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   └── migrate_to_s3.cpython-313.pyc
├── demo.ipynb
├── migrate_to_s3.py
├── photos.jsonl
├── shopify_image_generator
│   ├── README.md
│   ├── main.py
│   └── urls.txt
└── sql
    ├── distinct_recent_image_users.sql
    ├── find_user_ravenousdesign.sql
    ├── find_user_ravenousdesign_like.sql
    ├── find_user_thornecanyon.sql
    ├── least_nonzero_images.sql
    ├── recent_image_counts.sql
    └── usage_stats.sql
models/
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   ├── feedback.cpython-313.pyc
│   ├── image.cpython-313.pyc
│   ├── product.cpython-313.pyc
│   └── user.cpython-313.pyc
├── feedback.py
├── image.py
├── product.py
└── user.py
core
├── __pycache__
│   ├── config.cpython-313.pyc
│   └── security.cpython-313.pyc
├── config.py
└── security.py

37 directories, 152 files
```

```
# config.py
app
├── __init__.py
├── __pycache__
│   └── main.cpython-313.pyc
├── api
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── auth.cpython-313.pyc
│   │   ├── blog.cpython-313.pyc
│   │   ├── deps.cpython-313.pyc
│   │   ├── exception_handling.cpython-313.pyc
│   │   ├── exceptions.cpython-313.pyc
│   │   ├── feedback.cpython-313.pyc
│   │   ├── images.cpython-313.pyc
│   │   ├── page_routes.cpython-313.pyc
│   │   ├── products.cpython-313.pyc
│   │   ├── subscriptions.cpython-313.pyc
│   │   └── users.cpython-313.pyc
│   ├── auth.py
│   ├── feedback.py
│   ├── images.py
│   ├── products.py
│   ├── subscriptions.py
│   └── users.py
├── deps
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── tokens.cpython-313.pyc
│   │   └── users.cpython-313.pyc
│   ├── tokens.py
│   └── users.py
├── main.py
├── page_routes
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── account_pages.cpython-313.pyc
│   │   ├── app_pages.cpython-313.pyc
│   │   ├── auth_pages.cpython-313.pyc
│   │   ├── feedback_pages.cpython-313.pyc
│   │   └── landing_pages.cpython-313.pyc
│   ├── account_pages.py
│   ├── app_pages.py
│   ├── auth_pages.py
│   ├── feedback_pages.py
│   └── landing_pages.py
├── static
│   ├── css
│   │   ├── styles.css
│   │   └── tailwind.css
│   ├── export
│   ├── images
│   │   ├── amazon.png
│   │   ├── cattoy-gen.png
│   │   ├── etsy.png
│   │   ├── laptop-gen.png
│   │   ├── laptop.png
│   │   ├── link-preview.png
│   │   ├── notebook.png
│   │   ├── pip_cattoy.png
│   │   ├── pip_shoe.png
│   │   ├── pip_stanley.png
│   │   ├── pip_tote.png
│   │   ├── shopify.png
│   │   ├── tote-lifestyle2.png
│   │   ├── tote-lifestyle3.png
│   │   ├── tote.png
│   │   ├── wellness_product_input.jpg
│   │   └── wellness_product_lifestyle.png
│   ├── js
│   │   ├── main.js
│   │   ├── product_detail.js
│   │   ├── products_list.js
│   │   └── settings.js
│   └── optimize_images.py
├── templates
│   ├── account_pages
│   │   ├── purchase_cancel.html
│   │   ├── purchase_success.html
│   │   └── settings.html
│   ├── app_pages
│   │   ├── dashboard.html
│   │   ├── product_detail.html
│   │   └── products_list.html
│   ├── auth_pages
│   │   └── signin.html
│   ├── base.html
│   ├── feedback_pages
│   │   └── feedback.html
│   ├── index.html
│   ├── landing_pages
│   │   ├── dropshipping.html
│   │   └── image_editing.html
│   ├── partials
│   │   ├── add_product_modal.html
│   │   ├── feedback_form.html
│   │   ├── image_card.html
│   │   ├── image_list.html
│   │   ├── login_form.html
│   │   └── product_card.html
│   └── pricing.html
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-313.pyc
    │   ├── blog_utils.cpython-313.pyc
    │   ├── exception_handling.cpython-313.pyc
    │   ├── htmx_snippets.cpython-313.pyc
    │   ├── model_to_response_adapters.cpython-313.pyc
    │   └── username.cpython-313.pyc
    ├── exception_handling.py
    ├── htmx_snippets.py
    ├── model_to_response_adapters.py
    └── username.py
database
├── __pycache__
│   └── db.cpython-313.pyc
└── db.py
services
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   ├── gemini_integration.cpython-313.pyc
│   ├── image_storage_service.cpython-313.pyc
│   ├── lifestyle_image_service.cpython-313.pyc
│   ├── notification_service.cpython-313.pyc
│   ├── ntfy_integration.cpython-313.pyc
│   ├── stripe_integration.cpython-313.pyc
│   ├── user_creation_util.cpython-313.pyc
│   └── user_quota_service.cpython-313.pyc
├── gemini_integration.py
├── lifestyle_image_service.py
├── notification_service.py
├── ntfy_integration.py
├── storage
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── base.cpython-313.pyc
│   │   ├── hybrid.cpython-313.pyc
│   │   ├── local.cpython-313.pyc
│   │   └── s3.cpython-313.pyc
│   ├── base.py
│   ├── hybrid.py
│   ├── local.py
│   └── s3.py
├── stripe_integration.py
├── user_creation_util.py
└── user_quota_service.py
scripts
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   └── migrate_to_s3.cpython-313.pyc
├── demo.ipynb
├── migrate_to_s3.py
├── photos.jsonl
├── shopify_image_generator
│   ├── README.md
│   ├── main.py
│   └── urls.txt
└── sql
    ├── distinct_recent_image_users.sql
    ├── find_user_ravenousdesign.sql
    ├── find_user_ravenousdesign_like.sql
    ├── find_user_thornecanyon.sql
    ├── least_nonzero_images.sql
    ├── recent_image_counts.sql
    └── usage_stats.sql
models/
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-313.pyc
│   ├── feedback.cpython-313.pyc
│   ├── image.cpython-313.pyc
│   ├── product.cpython-313.pyc
│   └── user.cpython-313.pyc
├── feedback.py
├── image.py
├── product.py
└── user.py
core
├── __pycache__
│   ├── config.cpython-313.pyc
│   └── security.cpython-313.pyc
├── config.py
└── security.py

37 directories, 152 files
```

Okay, with all this laid out, please set up a minimal configuration for me for this project.
Before you make changes, give me a hyper-terse overview of what you think you should create.
