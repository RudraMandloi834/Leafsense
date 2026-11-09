import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "or": "Odia",
}


def generate_fallback_advice(diseases, language):
    """Generate basic fallback advice when Sarvam API is unavailable or unconfigured."""
    clean_diseases = [d.replace("___", " - ").replace("_", " ") for d in diseases]
    disease_text = ", ".join(clean_diseases)

    if language == "English":
        return f"""## Detected Diseases
{disease_text}

## 🌿 Agricultural Management Advisory

Early detection and targeted treatment are vital for managing crop pathology and preserving yield.

### Immediate Action Plan:
- 🔍 **Isolate Affected Plants**: Quarantine heavily symptomatic foliage to prevent pathogen spores from spreading.
- 🗑️ **Sanitation**: Prune and safely dispose of infected leaf tissue away from the field or compost pile.
- 💧 **Irrigation Management**: Avoid overhead sprinkler irrigation; keep foliage dry and water directly at the root zone.
- 🌬️ **Canopy Aeration**: Ensure proper plant spacing and pruning to promote good air circulation.
- 🧴 **Targeted Protection**: Apply appropriate organic bio-fungicides or approved chemical treatments suited for {clean_diseases[0] if clean_diseases else 'the detected disease'}.
- 👨‍🌾 **Local Extension Services**: Consult your regional Krishi Vigyan Kendra (KVK) or agricultural extension officer for season-specific recommendations.
"""
    else:
        return f"""## Detected Diseases / पता लगाई गई बीमारियाँ
{disease_text}

## 🌿 Agricultural Advisory / कृषि सलाह

We have detected plant disease symptoms. Follow these key management practices:
हमने पौधे की बीमारी का पता लगाया है। कृपया इन प्रमुख प्रबंधन प्रथाओं का पालन करें:

### Recommendations / सामान्य अनुशंसाएँ:
- 🔍 **Isolate Plants / पौधों को अलग करें**: Isolate diseased plants to prevent rapid field spread / प्रभावित पौधों को अलग करें ताकि बीमारी न फैले।
- 🗑️ **Remove Infected Foliage / संक्रमित भाग हटाएं**: Prune infected leaves immediately / संक्रमित पत्तियों को तुरंत हटाएं और नष्ट करें।
- 💧 **Watering Control / सिंचाई नियंत्रण**: Keep leaves dry during irrigation / पत्तियों को सूखा रखें और जड़ों में पानी दें।
- 👨‍🌾 **Expert Guidance / विशेषज्ञ परामर्श**: Consult your local agricultural officer / अपने स्थानीय कृषि विस्तार अधिकारी या केवीके से संपर्क करें।
"""


def get_advice(diseases, lang_code="en"):
    """
    Generate multilingual agricultural advice using Sarvam AI.
    Falls back gracefully if the API key is not configured or network fails.
    """
    language = LANGUAGES.get(lang_code, "English")

    # Read dynamically in case .env was modified after process startup
    api_key = os.getenv("SARVAM_API_KEY", "").strip()

    if not api_key:
        return generate_fallback_advice(diseases, language)

    clean_diseases = [d.replace("___", " - ").replace("_", " ") for d in diseases]
    disease_text = ", ".join(clean_diseases)

    prompt = f"""
Detected crop diseases: {disease_text}
Target Language: {language}

You are an expert plant pathologist and agronomist. Generate a practical, farmer-friendly disease diagnosis and management report in {language}.

For each detected disease, provide:
🦠 Cause (कारण)
🍃 Symptoms (लक्षण)
💊 Treatment (उपचार)
🛡️ Prevention (रोकथाम)

Followed by:
🌾 Integrated Management Strategy
⚠️ Critical Agricultural Warnings

Requirements:
- Write completely in {language}.
- Use simple, direct, farmer-friendly terms.
- Use clear bullet points and markdown headers (## and ###).
- Do not mention instructions, meta-commentary, or reasoning steps.
- Output only the final agricultural advisory.
"""

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-30b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert agronomist providing actionable farmer advice. "
                            "Output only the final report with clean markdown headers and bullet points. "
                            "Do not include thinking, reasoning, or meta commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2500,
                "temperature": 0.2,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"Sarvam AI API Error (Status {response.status_code}): {response.text[:200]}")
            return generate_fallback_advice(diseases, language)

        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"].get("content")
            if content:
                # Strip thinking tags if generated by reasoning models
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content

        return generate_fallback_advice(diseases, language)

    except Exception as e:
        print(f"Sarvam AI advisory call exception: {e}")
        return generate_fallback_advice(diseases, language)
