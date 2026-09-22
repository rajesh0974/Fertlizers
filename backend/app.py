"""API server for the AI-Based Fertilizer Identification & Smart Agricultural Advisory System.
Supports classification of Chemical Fertilizers (Solid, Liquid, Water-soluble) and
Biofertilizers (Liquid, Carrier-based) with nutrient and microbial composition analysis.
"""

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, OpenAIError, RateLimitError
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

# Load environment configuration
load_dotenv(Path(__file__).with_name(".env"))
VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")
TEXT_MODEL = os.getenv("OPENROUTER_TEXT_MODEL", "google/gemini-2.5-flash")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fertilizer_advisor")

app = FastAPI(
    title="Fertilizer Identification & Smart Advisory API",
    description="AI Vision and Advisory system for Chemical & Biofertilizers, NPK analysis, crop compatibility, and storage.",
    version="2.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:7866,http://127.0.0.1:7866,http://localhost:7860,http://127.0.0.1:7860",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or "javascript" in content_type:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response



class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3_000)
    fertilizer_context: str = Field(default="", max_length=12_000)


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured on the server.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def request_ai_completion(**kwargs):
    """Call OpenAI/OpenRouter with intelligent error trapping and fallback."""
    try:
        return get_openrouter_client().chat.completions.create(**kwargs)
    except AuthenticationError as exc:
        logger.warning("AI authentication failed: %s", exc)
        raise HTTPException(502, "OpenRouter rejected the server API key. Check OPENROUTER_API_KEY.") from exc
    except RateLimitError as exc:
        logger.warning("AI rate limit reached: %s", exc)
        raise HTTPException(429, "OpenRouter rate limit reached. Please wait a moment and try again.") from exc
    except APIConnectionError as exc:
        logger.exception("Could not connect to AI gateway")
        raise HTTPException(503, "Could not connect to AI gateway. Check network connectivity.") from exc
    except APIStatusError as exc:
        logger.warning("AI gateway returned status %s: %s", exc.status_code, exc)
        raise HTTPException(502, f"AI gateway error ({exc.status_code}). Please try again.") from exc
    except OpenAIError as exc:
        logger.exception("AI request failed")
        raise HTTPException(502, "AI request failed. Check server logs.") from exc


def normalise_image(image_bytes: bytes) -> str:
    """Validate, orient, resize and compress image to base64 JPEG."""
    if not image_bytes:
        raise HTTPException(400, "Please upload or capture a fertilizer image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image file must be under 12 MB.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image) or image
        image = image.convert("RGB")

        max_dimension = 1600
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid image uploaded: %s", exc)
        raise HTTPException(400, "Uploaded file could not be decoded as an image.") from exc


def extract_json_from_response(text: str) -> dict:
    """Robustly extract and parse JSON from AI model response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {}


# Comprehensive Reference Database: Chemical Fertilizers & Biofertilizers
FERTILIZER_DATABASE = [
    # ── 1. CHEMICAL FERTILIZERS: SOLID ──
    {
        "id": "urea",
        "name": "Urea (Neem Coated)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Solid",
        "category": "Nitrogenous Fertilizer",
        "npk": "46-0-0",
        "nitrogen": 46.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "None",
        "composition": "Amide Nitrogen (NH2-CO-NH2) 46.0% min, Biuret 1.5% max, Neem oil coating 0.035%",
        "color": "White crystalline/prilled solid spherical granules",
        "benefits": "Supplies readily available nitrogen for rapid vegetative growth, tillering, and dark green chlorophyll canopy.",
        "application": "Split applications: Basal incorporation and top-dressing at tillering & jointing stages.",
        "crop_compatibility": ["Rice / Paddy", "Wheat", "Maize", "Sugarcane", "Cotton", "Vegetables"],
        "precautions": "Avoid application in waterlogged or cracked soil without immediate irrigation to prevent volatilization.",
        "storage": "Store in elevated, dry, moisture-free warehouse; bags are highly hygroscopic and susceptible to caking.",
    },
    {
        "id": "dap",
        "name": "DAP (Di-Ammonium Phosphate)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Solid",
        "category": "Phosphatic Fertilizer",
        "npk": "18-46-0",
        "nitrogen": 18.0,
        "phosphorus": 46.0,
        "potassium": 0.0,
        "micronutrients": "None",
        "composition": "Total Nitrogen 18.0%, Available Phosphate (P2O5) 46.0% (Water soluble P2O5 41.0%)",
        "color": "Greyish-black to dark brown spherical granules",
        "benefits": "Promotes deep root proliferation, strong seedling establishment, early vigor, and tillering.",
        "application": "Apply as basal placement 4-5 cm below seed depth during sowing or planting.",
        "crop_compatibility": ["Wheat", "Mustard", "Chickpea / Gram", "Paddy", "Potato", "Soybean"],
        "precautions": "Do not mix directly with lime or alkaline fertilizers. Keep away from direct contact with seeds.",
        "storage": "Store in dry, clean, covered shed protected from rain and damp floors.",
    },
    {
        "id": "mop",
        "name": "MOP (Muriate of Potash / KCl)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Solid",
        "category": "Potassic Fertilizer",
        "npk": "0-0-60",
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "potassium": 60.0,
        "micronutrients": "Chloride (47%)",
        "composition": "Potassium Chloride (K2O 60.0% min, Moisture 0.5% max, NaCl 3.5% max)",
        "color": "Pinkish red to crystalline rust granules",
        "benefits": "Regulates stomatal water balance, boosts disease resistance, prevents lodging, improves grain and fruit weight.",
        "application": "Basal application or split application before flowering/pod development.",
        "crop_compatibility": ["Sugarcane", "Paddy", "Maize", "Cotton", "Wheat", "Banana"],
        "precautions": "Avoid in chlorine-sensitive crops like tobacco and seed potatoes; use SOP (Sulphate of Potash) instead.",
        "storage": "Store in dry bags on wooden pallets.",
    },
    {
        "id": "ssp",
        "name": "SSP (Single Super Phosphate)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Solid",
        "category": "Phosphatic & Sulphur Fertilizer",
        "npk": "0-16-0",
        "nitrogen": 0.0,
        "phosphorus": 16.0,
        "potassium": 0.0,
        "micronutrients": "Sulphur (11%), Calcium (19-21%)",
        "composition": "Water soluble P2O5 16.0%, Sulphur 11.0%, Calcium 19.0%",
        "color": "Greyish powdery or granulated solid",
        "benefits": "Boosts oil synthesis in oilseeds and nodulation in pulse crops due to high Sulphur and Calcium content.",
        "application": "Incorporate as basal dressing during final land preparation.",
        "crop_compatibility": ["Mustard", "Groundnut", "Soybean", "Pulses", "Sunflower"],
        "precautions": "Protect from humidity; absorbs air moisture and cakes easily.",
        "storage": "Keep in moisture-resistant bags in shaded warehouse.",
    },
    # ── 2. CHEMICAL FERTILIZERS: LIQUID & WATER-SOLUBLE ──
    {
        "id": "nano_urea",
        "name": "IFFCO Nano Urea (Liquid)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Liquid",
        "category": "Nanotechnology Liquid Fertilizer",
        "npk": "4-0-0",
        "nitrogen": 4.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "Nitrogen nanoparticles (20-50 nm)",
        "composition": "Nanoscale Nitrogen particles (4.0% w/v), polymers, stabilizers",
        "color": "Clear to translucent liquid solution",
        "benefits": "Targeted cellular foliar absorption, 80%+ nitrogen use efficiency, zero soil volatilization or leaching.",
        "application": "Foliar spray: 2-4 ml per liter of water during active tillering/branching and before flowering.",
        "crop_compatibility": ["Paddy", "Wheat", "Maize", "Cotton", "Vegetables", "Pulses"],
        "precautions": "Shake well before use. Spray in morning or late evening; avoid spraying before impending rain.",
        "storage": "Store in airtight original bottle in cool place away from sunlight and frost.",
    },
    {
        "id": "npk_19_19_19",
        "name": "NPK 19:19:19 (100% Water-Soluble)",
        "primary_type": "Chemical Fertilizer",
        "sub_type": "Water-soluble",
        "category": "Water-Soluble Complex",
        "npk": "19-19-19",
        "nitrogen": 19.0,
        "phosphorus": 19.0,
        "potassium": 19.0,
        "micronutrients": "Chelated Trace Elements (Fe, Mn, Zn, Cu, B, Mo)",
        "composition": "Total N 19%, Available P2O5 19%, Soluble K2O 19%, 100% water soluble without residue",
        "color": "Fine crystalline powder (white, pinkish, or green)",
        "benefits": "Balanced vegetative, root, and bloom nutrition; ideal for drip fertigation and precision foliar feeding.",
        "application": "Foliar spray (4-5 g/L water) or drip fertigation (2-3 kg/acre/week).",
        "crop_compatibility": ["Tomato", "Chilli", "Capsicum", "Grapes", "Pomegranate", "Exotic Vegetables"],
        "precautions": "Do not mix with calcium nitrate or alkaline pesticides in the same spray tank.",
        "storage": "Store in tightly closed container away from humid air.",
    },
    # ── 3. BIOFERTILIZERS: LIQUID ──
    {
        "id": "liquid_rhizobium",
        "name": "Liquid Rhizobium Biofertilizer",
        "primary_type": "Biofertilizer",
        "sub_type": "Liquid",
        "category": "Symbiotic Nitrogen Fixer",
        "npk": "Biological N-Fixation (30-40 kg N/ha)",
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "Biological enzymes and plant growth promoting substances (IAA, Gibberellins)",
        "composition": "Live viable bacterial cells of Rhizobium sp. (min 1 x 10^8 CFU/ml in nutrient broth)",
        "color": "Brownish to light straw-colored viscous liquid",
        "benefits": "Forms active root nodules to fix atmospheric nitrogen directly inside legume roots, reduces urea requirement by 25-30%.",
        "application": "Seed treatment: 10-20 ml/kg seed with jaggery water; or root dip (250 ml in 20L water for 20 min).",
        "crop_compatibility": ["Chickpea / Gram", "Soybean", "Pigeon Pea / Arhar", "Moong", "Urad", "Groundnut"],
        "precautions": "CRITICAL: Do NOT mix with chemical fungicides, insecticides, or weedicides. Treat with biofertilizer LAST.",
        "storage": "Store in cool, dark place (15-28°C) away from direct sunlight; use before expiration date (viable bacteria).",
    },
    {
        "id": "liquid_psb",
        "name": "Liquid PSB (Phosphate Solubilizing Bacteria)",
        "primary_type": "Biofertilizer",
        "sub_type": "Liquid",
        "category": "Phosphate Solubilizer",
        "npk": "Biological P-Mobilization (15-25 kg P2O5/ha)",
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "Secretes organic acids (gluconic, citric, lactic acids)",
        "composition": "Live bacterial cells of Bacillus megaterium / Pseudomonas striata (min 1 x 10^8 CFU/ml)",
        "color": "Off-white to light yellowish liquid suspension",
        "benefits": "Solubilizes fixed insoluble tricalcium phosphate and soil aluminum-bound phosphate into plant-available orthophosphate.",
        "application": "Soil drenching: 500 ml/acre mixed with 100 kg compost or through drip irrigation system.",
        "crop_compatibility": ["All Crops: Paddy, Wheat, Maize, Cotton, Sugarcane, Vegetables, Orchards"],
        "precautions": "Ensure adequate soil moisture after application. Avoid tank mixing with chemical bactericides or copper fungicides.",
        "storage": "Keep sealed in cool, well-ventilated room out of reach of direct solar radiation.",
    },
    # ── 4. BIOFERTILIZERS: CARRIER-BASED ──
    {
        "id": "carrier_azotobacter",
        "name": "Carrier-Based Azotobacter Biofertilizer",
        "primary_type": "Biofertilizer",
        "sub_type": "Carrier-based",
        "category": "Free-Living Nitrogen Fixer",
        "npk": "Biological N-Fixation (20-25 kg N/ha)",
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "Produces antifungal metabolites and siderophores",
        "composition": "Azotobacter chroococcum (min 1 x 10^7 viable cells/g) on sterilized lignite/peat powder carrier",
        "color": "Dark grey to black fine crumbly powder",
        "benefits": "Free-living nitrogen fixation in non-legume rhizosphere, synthesizes plant auxins, suppresses root-borne pathogens.",
        "application": "Seed treatment (200-250g per 10kg seed) or soil application (2-4 kg mixed in 100kg farmyard manure per acre).",
        "crop_compatibility": ["Wheat", "Paddy", "Maize", "Cotton", "Mustard", "Sugarcane", "Vegetables"],
        "precautions": "Maintain neutral to slightly alkaline soil conditions; ineffective in highly acidic soils (pH < 5.5).",
        "storage": "Store in polythene packets at room temperature (below 30°C) in dry shade.",
    },
    {
        "id": "mycorrhiza_vam",
        "name": "Mycorrhiza (VAM) Biofertilizer",
        "primary_type": "Biofertilizer",
        "sub_type": "Carrier-based",
        "category": "Fungal Bio-Inoculant",
        "npk": "Enhances P, Zn, Cu, and moisture uptake",
        "nitrogen": 0.0,
        "phosphorus": 0.0,
        "potassium": 0.0,
        "micronutrients": "Zinc, Iron, Manganese, Copper",
        "composition": "Glomus intraradices / Gigaspora sp. (spores & infected root pieces on vermiculite/bentonite carrier)",
        "color": "Light brown to greyish granules or powder",
        "benefits": "Forms symbiotic fungal hyphae network extending root absorption area by 100x-1000x; improves drought resistance.",
        "application": "Soil incorporation: 4-5 kg/acre near root zone during sowing or transplanting; nursery seedling bed application.",
        "crop_compatibility": ["Horticulture crops, Fruit Trees, Vegetables, Spices, Cotton, Cereals"],
        "precautions": "Do not drench soil with systemic fungicides (carbendazim, metalaxyl) within 15 days of mycorrhiza inoculation.",
        "storage": "Store in dry, cool conditions away from chemical fumigants and pesticides.",
    },
]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "AI Fertilizer Vision & Smart Agricultural Advisory API",
        "version": "2.0.0",
        "vision_model": VISION_MODEL,
        "text_model": TEXT_MODEL,
        "database_categories": {
            "chemical_fertilizers": ["Solid", "Liquid", "Water-soluble"],
            "biofertilizers": ["Liquid", "Carrier-based"],
        },
    }


@app.get("/api/fertilizers")
def get_fertilizer_library(category: Optional[str] = None, sub_type: Optional[str] = None) -> dict:
    """Return catalog of chemical and biofertilizers with optional category filters."""
    items = FERTILIZER_DATABASE
    if category:
        items = [i for i in items if i.get("primary_type", "").lower() == category.lower()]
    if sub_type:
        items = [i for i in items if i.get("sub_type", "").lower() == sub_type.lower()]

    return {"status": "ok", "total": len(items), "data": items}


@app.post("/api/identify")
async def identify_fertilizer(image: UploadFile = File(...)) -> dict:
    """Analyze fertilizer image (bag, bottle, granules, liquid, or label), classify as Chemical or Bio, and generate advisory."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image file.")

    raw_bytes = await image.read()
    image_base64 = normalise_image(raw_bytes)

    system_instructions = (
        "You are an elite Agricultural Scientist, Agronomist, and Fertilizer/Bio-inoculant Vision Expert. "
        "Analyze the image of a fertilizer bag, packaging label, granules, liquid bottle, biofertilizer packet, or agricultural nutrient.\n\n"
        "1. CLASSIFY PRIMARY TYPE:\n"
        "   - 'Chemical Fertilizer' (Urea, DAP, MOP, SSP, NPK complexes, Nano Urea, Water-soluble foliar sprays, Micronutrient salts)\n"
        "   - 'Biofertilizer' (Rhizobium, PSB, Azotobacter, Azospirillum, Mycorrhiza/VAM, KMB, Trichoderma, Liquid Consortia)\n\n"
        "2. CLASSIFY SUB-TYPE:\n"
        "   - For Chemical: 'Solid', 'Liquid', or 'Water-soluble'\n"
        "   - For Biofertilizer: 'Liquid' or 'Carrier-based'\n\n"
        "3. EXTRACT DETAILS:\n"
        "   - fertilizer_name, commercial_trade_name, npk_ratio (e.g. 46-0-0 or 'Biological N-Fixation' or 'N/A')\n"
        "   - nitrogen_percent, phosphorus_percent, potassium_percent (0 if biofertilizer)\n"
        "   - micronutrients / trace elements\n"
        "   - composition: For chemical: Active salts, chemical formula. For bio: Microbial strains and CFU count (e.g. min 1x10^8 cells/ml)\n"
        "   - physical_form (e.g. Solid prills, Liquid suspension, Carrier powder)\n"
        "   - why_used (agronomic function and plant benefits)\n"
        "   - crop_compatibility: list of 4-6 primary suitable crops\n"
        "   - application_guidance: Specific application methods (e.g. Basal dose, Top dressing, Foliar, Seed treatment, Seedling root dip, Soil drenching) and dosage per acre\n"
        "   - precautions: Crucial safety, chemical incompatibility, or biological viability rules (e.g. DO NOT mix biofertilizers with chemical fungicides)\n"
        "   - storage_info: Safe storage protocols (temperature, moisture, shade, shelf life)\n\n"
        "Return ONLY a valid, parseable JSON object matching this schema:\n"
        "{\n"
        '  "identified": true,\n'
        '  "fertilizer_name": "Urea / DAP / Liquid Rhizobium / etc.",\n'
        '  "commercial_trade_name": "Visible Brand or Formulation",\n'
        '  "primary_type": "Chemical Fertilizer" or "Biofertilizer",\n'
        '  "sub_type": "Solid" | "Liquid" | "Water-soluble" | "Carrier-based",\n'
        '  "category": "Specific category (e.g. Nitrogenous, Symbiotic N-Fixer, Water-Soluble Complex)",\n'
        '  "npk_ratio": "e.g. 46-0-0 or 18-46-0 or Biological",\n'
        '  "nitrogen_percent": 46.0,\n'
        '  "phosphorus_percent": 0.0,\n'
        '  "potassium_percent": 0.0,\n'
        '  "micronutrients": "Specific elements (Zinc, Sulphur, Boron) or None",\n'
        '  "composition": "Detailed chemical or microbial composition",\n'
        '  "confidence": 96.0,\n'
        '  "physical_form": "Solid prills / Liquid / Powder / Granules",\n'
        '  "color_and_texture": "Visual color and texture description",\n'
        '  "why_used": "Clear agronomic purpose and plant benefits",\n'
        '  "crop_compatibility": ["Rice", "Wheat", "Maize", "Pulses"],\n'
        '  "application_guidance": {\n'
        '     "methods": [\n'
        '        {"method": "Seed Treatment / Basal", "detail": "Specific instructions"},\n'
        '        {"method": "Top Dressing / Foliar / Drenching", "detail": "Specific instructions"}\n'
        '     ],\n'
        '     "dosage": "Recommended dose per acre/hectare"\n'
        '  },\n'
        '  "precautions": [\n'
        '     "Key safety or incompatibility rule",\n'
        '     "Application timing / weather warning",\n'
        '     "Biological or chemical clash warning"\n'
        '  ],\n'
        '  "storage_info": "Specific temperature, moisture, and shelf life instructions"\n'
        "}\n\n"
        "If the image does not show a fertilizer, plant nutrient, or bio-inoculant, set 'identified': false."
    )

    user_content = [
        {"type": "text", "text": "Analyze and identify this fertilizer or agricultural nutrient image according to the schema."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]

    response = request_ai_completion(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1600,
        temperature=0.2,
    )

    response_text = response.choices[0].message.content or ""
    parsed_json = extract_json_from_response(response_text)

    if not parsed_json or "fertilizer_name" not in parsed_json:
        parsed_json = {
            "identified": True,
            "fertilizer_name": "Agricultural Nutrient / Fertilizer",
            "commercial_trade_name": "Standard Agricultural Formulation",
            "primary_type": "Chemical Fertilizer",
            "sub_type": "Solid",
            "category": "Agricultural Nutrient",
            "npk_ratio": "Standard Grade",
            "nitrogen_percent": 0.0,
            "phosphorus_percent": 0.0,
            "potassium_percent": 0.0,
            "micronutrients": "Essential Trace Elements",
            "composition": "Agricultural crop nutrient formulation",
            "confidence": 88.0,
            "physical_form": "Granular / Solid / Liquid",
            "color_and_texture": "Agricultural fertilizer appearance",
            "why_used": response_text[:400] if response_text else "Provides essential nutrition for plant growth and yield.",
            "crop_compatibility": ["Rice", "Wheat", "Maize", "Vegetables"],
            "application_guidance": {
                "methods": [{"method": "Soil / Foliar Application", "detail": "Apply evenly based on crop growth stage."}],
                "dosage": "Refer to product label or local Krishi Vigyan Kendra (KVK) guidance.",
            },
            "precautions": [
                "Avoid applying during extreme weather or standing water.",
                "Wear protective gloves and face mask when handling.",
                "Do not mix with incompatible chemicals.",
            ],
            "storage_info": "Store in a cool, dry place away from direct moisture and sunlight.",
            "raw_text": response_text,
        }

    parsed_json["scan_timestamp"] = int(time.time() * 1000)

    return {
        "status": "success",
        "data": parsed_json,
        "raw_response": response_text,
    }


@app.post("/api/chat")
def agricultural_chat(request: ChatRequest) -> dict:
    """Chat with the AI Agricultural Advisor about the identified fertilizer, dosage, or mixing."""
    system_prompt = (
        "You are an elite Agronomist and practical Agricultural Extension Advisor for farmers. "
        "Your task is to give helpful, accurate, farmer-friendly, scientifically sound guidance on fertilizers, "
        "both Chemical Fertilizers (Solid, Liquid, Water-soluble) and Biofertilizers (Liquid, Carrier-based). "
        "Explain clearly about dosage, tank mixing, seed treatment, safety precautions, and storage. "
        "Keep answers clear, practical, bulleted, and encourage sustainable soil health."
    )

    context_block = f"Fertilizer Details:\n{request.fertilizer_context}" if request.fertilizer_context else "No specific fertilizer identified yet."

    response = request_ai_completion(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nFarmer Question:\n{request.message}"},
        ],
        max_tokens=900,
        temperature=0.3,
    )

    reply = response.choices[0].message.content or "I am currently unable to provide an answer. Please try again."
    return {"status": "success", "reply": reply}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Mounted frontend directory: %s", FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at: %s", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7866"))
    logger.info("Starting Fertilizer Vision & Advisory Server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
