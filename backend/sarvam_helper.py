import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

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
    """Generate basic fallback advice when API is unavailable"""
    disease_text = ", ".join(diseases)

    if language == "English":
        return f"""## Detected Diseases
{disease_text}

## ⚠️ Translation Service Temporarily Unavailable

We've detected the above disease(s) in your plant. Please consult with a local agricultural expert for detailed treatment recommendations.

### General Recommendations:
- 🔍 Isolate affected plants to prevent spread
- 🗑️ Remove and destroy infected plant parts
- 💧 Ensure proper watering (avoid overwatering)
- 🌬️ Improve air circulation around plants
- 🧴 Consider applying appropriate fungicides or treatments
- 👨‍🌾 Consult local agricultural extension officers

### Important:
Early detection and treatment are crucial for disease management. Please seek professional guidance for your specific situation.
"""
    else:
        return f"""## Detected Diseases / पता लगाई गई बीमारियाँ
{disease_text}

## ⚠️ Translation service temporarily unavailable / अनुवाद सेवा अस्थायी रूप से अनुपलब्ध

We've detected plant disease(s). Please consult a local agricultural expert.
हमने पौधे की बीमारी का पता लगाया है। कृपया स्थानीय कृषि विशेषज्ञ से परामर्श करें।

### General Advice / सामान्य सलाह:
- Isolate affected plants / प्रभावित पौधों को अलग करें
- Remove infected parts / संक्रमित भागों को हटाएं
- Improve air circulation / हवा का संचार बेहतर करें
- Consult agricultural expert / कृषि विशेषज्ञ से परामर्श करें
"""


def get_advice(diseases, lang_code):

    language = LANGUAGES.get(lang_code, "English")

    # Check if API key is available
    if not API_KEY or API_KEY == "":
        print("⚠️ Warning: SARVAM_API_KEY not found in environment")
        return generate_fallback_advice(diseases, language)

    disease_text = ", ".join(diseases)

    prompt = f"""
Detected diseases:
{disease_text}

Language: {language}

Generate agricultural recommendations.

For each disease provide:

🦠 Cause
🍃 Symptoms
💊 Treatment
🛡️ Prevention

After all diseases provide:

🌾 Management Strategy
⚠️ Important Notes

Requirements:
- Use simple farmer-friendly language.
- Use bullet points.
- Give practical recommendations.
- Avoid scientific jargon whenever possible.
- Respond completely in {language}.
- Output only the final report.
"""

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
    "model": "sarvam-30b",
    "messages": [
    {
        "role": "system",
        "content": """
You are an agricultural expert.

Return ONLY the final farmer-facing report.

Never:
- Explain your reasoning
- Explain your thinking
- Say 'Now let me translate'
- Say 'I've done this'
- Say 'I've included'
- Output checkmarks (✓)
- Mention instructions
- Mention requirements
- Mention formatting rules

Start directly with the disease report.
"""
    },
    {
        "role": "user",
        "content": prompt
    }
],
        
                "max_tokens": 3500,  # Reduced for faster, more focused response
                "temperature": 0.1,  # Add temperature for more deterministic output
            },
            timeout=60,
        )

        data = response.json()

        # Debug: Print response status and structure
        print(f"📡 Sarvam AI Status: {response.status_code}")
        if "choices" in data:
            print(f"✓ Response has 'choices' key with {len(data['choices'])} items")
        else:
            print(f"❌ Response missing 'choices' key. Keys: {list(data.keys())}")

        # Check if response has the expected structure
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0]["message"]

            # Try to get content from either 'content' or 'reasoning_content'
            content = message.get("content")

            if content:
                content = re.sub(
                    r"<think>.*?</think>",
                    "",
                    content,
                    flags=re.DOTALL
                ).strip()

                content = re.sub(
                    r"(?i)i've done this.*", "", content)
            
            # Debug: Show what we got
            if content:
                print(f"📝 Got 'content' field ({len(content)} chars)")
                # Check if content contains thinking keywords
                if any(keyword in content[:500] for keyword in ["Deconstruct", "Analyze", "Gather Information", "Translate and Simplify"]):
                    print(f"⚠️ Content appears to contain reasoning/thinking process")
            else:
                print(f"❌ 'content' field is None")

            # If content is None, try reasoning_content but extract only the clean output
            if content is None and message.get("reasoning_content"):
                reasoning = message.get("reasoning_content")
                
                print(f"⚙️ Attempting to extract from reasoning_content ({len(reasoning)} chars)")
                
                # Strategy: Extract only lines that look like actual disease advice
                # Skip lines with strong meta-commentary indicators
                strong_meta_keywords = [
                    "Let me", "I need to", "I'll", "I think", "Actually, let me", 
                    "Wait,", "Now let me", "Let me check", "I notice that",
                    "The user wants", "The user wrote", "Requirements:", 
                    "CRITICAL INSTRUCTIONS"
                ]
                
                lines = reasoning.split("\n")
                clean_lines = []
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Skip empty lines
                    if not stripped:
                        clean_lines.append(line)
                        continue
                    
                    # Check if this line has strong meta-commentary
                    has_strong_meta = any(keyword in line for keyword in strong_meta_keywords)
                    
                    # Check if line is template/placeholder (has brackets with keywords)
                    is_template = (
                        ("[cause]" in stripped or "[symptoms]" in stripped or 
                         "[treatment]" in stripped or "[prevention]" in stripped or
                         "[strategy]" in stripped or "[notes]" in stripped or
                         "[Disease" in stripped) and len(stripped) < 100
                    )
                    
                    # Keep the line if it's not strong meta and not a template
                    if not has_strong_meta and not is_template:
                        clean_lines.append(line)
                
                # Further cleanup: Remove leading meta-commentary
                # Find first line with disease heading or emoji
                start_idx = 0
                for i, line in enumerate(clean_lines):
                    if (line.strip().startswith("##") or 
                        any(emoji in line[:10] for emoji in ['🦠', '🍃', '💊', '🛡️', '🌾', '⚠️', '🎯'])):
                        start_idx = i
                        break
                
                clean_lines = clean_lines[start_idx:]
                
                if clean_lines and len(clean_lines) > 3:  # Must have some content
                    content = "\n".join(clean_lines).strip()
                    print(f"✓ Extracted {len(clean_lines)} lines from reasoning_content")
                else:
                    print(f"❌ Could not extract clean answer (only {len(clean_lines)} lines)")
                    print(f"Sample of reasoning_content: {reasoning[:500]}...")
                    return generate_fallback_advice(diseases, language)

            # Check if content is valid before processing
            if content is None or content == "":
                print(f"❌ API returned None/empty content. Full response: {data}")
                return generate_fallback_advice(diseases, language)
            
            # If content has thinking process, try to extract clean output
            if any(keyword in content[:500] for keyword in ["Deconstruct", "Analyze", "Gather Information", "Translate and Simplify", "Requirements:"]):
                print(f"⚙️ Applying extraction to remove thinking process")
                original_content = content
                
                # Apply the same extraction logic
                strong_meta_keywords = [
                    "1. Deconstruct", "2. Analyze", "3. Gather", "4. Translate", "5. Create",
                    "Let me", "I need to", "I'll", "Actually, let me", 
                    "Requirements:", "CRITICAL INSTRUCTIONS", "The user wants", "Topic:"
                ]
                
                lines = content.split("\n")
                clean_lines = []
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Skip empty lines at the start, but keep them later for formatting
                    if not stripped and not clean_lines:
                        continue
                    
                    # Check if this line has strong meta-commentary
                    has_strong_meta = any(keyword in line for keyword in strong_meta_keywords)
                    
                    # Check if line starts with numbered list markers (thinking steps)
                    is_numbered_step = stripped and stripped[0].isdigit() and stripped[1:3] in ['. ', ') ']
                    
                    # Keep the line if it's not meta
                    if not has_strong_meta and not is_numbered_step:
                        clean_lines.append(line)
                
                # Find first line with disease heading or emoji
                start_idx = 0
                for i, line in enumerate(clean_lines):
                    if (line.strip().startswith("##") or 
                        any(emoji in line[:10] for emoji in ['🦠', '🍃', '💊', '🛡️', '🌾', '⚠️', '🎯', '🍎'])):
                        start_idx = i
                        break
                
                clean_lines = clean_lines[start_idx:]
                
                if clean_lines and len(clean_lines) > 3:
                    content = "\n".join(clean_lines).strip()
                    print(f"✓ Extracted {len(clean_lines)} clean lines (removed thinking process)")
                else:
                    print(f"⚠️ Extraction resulted in too few lines ({len(clean_lines)}), using original")
                    content = original_content

            print(f"✓ Final content ready ({len(content)} chars)")

            # Remove thinking tags if present
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()
            
            # Remove common meta-phrases
            remove_phrases = [
                "Now, let me translate",
                "✓",
                "I've done this",
                "I've included",
                "Simple farmer-friendly language",
                "Detailed explanations",
                "No reasoning or instructions",
                "Requirements:"
            ]

            for phrase in remove_phrases:
                if phrase in content:
                    content = content.split(phrase)[0].strip()
            
            return content
        elif "error" in data:
            # API returned an error
            error_msg = data.get("error", {}).get("message", "Unknown API error")
            print(f"❌ Sarvam AI API Error: {error_msg}")
            return generate_fallback_advice(diseases, language)
        else:
            # Unexpected response format
            print(f"❌ Unexpected API response format: {data}")
            return generate_fallback_advice(diseases, language)

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error calling Sarvam AI: {e}")
        return generate_fallback_advice(diseases, language)
    except Exception as e:
        print(f"❌ Error in get_advice: {e}")
        return generate_fallback_advice(diseases, language)
