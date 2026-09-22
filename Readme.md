# 🌱 AI-Based Fertilizer Identification & Smart Agricultural Advisory System Using Image Processing

An end-to-end computer vision and agronomic advisory system designed for farmers and agronomists, integrated into the **Krishi-AI** ecosystem.

## 🌟 Features

- 📷 **Dual-Mode Fertilizer Capture**: Live camera viewfinder with rear/front switching, target reticle, instant snapshot, and native camera file fallback.
- 🖼️ **Image Preprocessing**: Auto EXIF orientation, noise filtering, dynamic contrast, and resolution normalization.
- 🤖 **AI-Driven Identification**: Deep multimodal vision models identify fertilizer packaging, brands, crystalline prills/granules, and nutrient grades.
- 🧪 **Chemical & Biofertilizers Classification**: Automatic distinction between Chemical Fertilizers (Solid, Liquid, Water-soluble) and Biofertilizers (Liquid, Carrier-based).
- 🔬 **NPK & Microbial Analysis**: Visual progress meters for Nitrogen (N), Phosphorus (P₂O₅), and Potassium (K₂O), plus secondary/trace elements and active microbial CFU counts.
- 🌾 **Crop Compatibility & Application Guidance**: Direct application guidance (Basal, Top Dressing, Foliar, Seed Treatment, Root Dip).
- ⚠️ **Precautions & Safe Storage**: Chemical incompatibility warnings, biofertilizer viability protection, and safe storage instructions.
- 💬 **Interactive AI Agri-Expert Chat**: Context-aware conversational assistant to ask dosage, timing, and tank-mixing questions.
- 📜 **Scan History**: Local history storage with image thumbnails and instant reload.
- 📚 **Built-in Fertilizer Catalog**: Quick lookup for standard chemical and biofertilizers.

---

## 🚀 Running the Project

### 1. Requirements & Setup
Make sure Python 3.10+ is installed.

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start Backend & Web Server
```bash
python app.py
```
The server will launch on port **7866**:
- **Application URL**: `http://localhost:7866`
- **Interactive API Docs**: `http://localhost:7866/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck and model status |
| `GET` | `/api/fertilizers` | Reference catalog of standard agricultural fertilizers |
| `POST` | `/api/identify` | Multipart image upload for AI identification & advisory |
| `POST` | `/api/chat` | Agricultural advisory chatbot with fertilizer context |
