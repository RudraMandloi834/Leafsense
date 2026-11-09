"""
LeafSense - Advanced Reasoning, Intelligent Research and Agricultural Knowledge Assistant
========================================================================================
Coordinates:
1. Plant disease classification predictions (from LeafSense PyTorch ensemble).
2. Farmer crop, variety, stage, field, and regional location context.
3. Multi-intent agricultural classification (15 categories).
4. Verified agricultural document retrieval & ranking (from LeafSense RAG & ChromaDB).
5. Differential disease analysis across confusing plant pathogens.
6. Multi-turn reasoning & empathetic generation via IBM Granite (ibm/granite-4-h-small).
7. Strict agricultural safety, chemical guardrails, and transparent uncertainty.
"""

import re
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from watsonx_service import watsonx_service, WatsonxServiceError
from rag_service import rag_service
from utils import format_disease_name

logger = logging.getLogger("leafsense.agent")

STANDARD_DISCLAIMER_EN = (
    "Disclaimer: LeafSense recommendations are AI-assisted preliminary advisories grounded "
    "in agricultural reference literature. They do not replace physical inspection by local "
    "agricultural extension officers or Krishi Vigyan Kendra (KVK) scientists. Always follow "
    "official CIB&RC label claims and local agricultural guidelines."
)

STANDARD_DISCLAIMER_HI = (
    "अस्वीकरण: LeafSense की सिफारिशें कृषि संदर्भ दस्तावेजों पर आधारित एआई-सहायक प्रारंभिक सलाह हैं। "
    "ये स्थानीय कृषि विस्तार अधिकारियों या कृषि विज्ञान केंद्र (KVK) के वैज्ञानिकों के प्रत्यक्ष फील्ड "
    "निरीक्षण का विकल्प नहीं हैं। किसी भी कृषि रसायन का उपयोग करने से पहले हमेशा आधिकारिक दिशानिर्देशों का पालन करें।"
)

# ----------------- PROGRESSIVE FARMER PROFILE SCHEMA -----------------
DEFAULT_FARMER_PROFILE: Dict[str, Optional[str]] = {
    "crop": None,                  # e.g., "Tomato"
    "location": None,              # e.g., "Nashik, Maharashtra"
    "variety": None,               # e.g., "Abhinav (Hybrid)"
    "plant_age": None,             # e.g., "35 days after transplanting"
    "growth_stage": None,          # e.g., "Flowering / Early fruit"
    "symptoms": None,              # e.g., "Concentric rings with yellow halo on lower leaves"
    "first_noticed": None,         # e.g., "3 days ago after rain"
    "is_spreading": None,          # e.g., "Spreading rapidly upward"
    "affected_percentage": None,   # e.g., "Around 15-20% of plot"
    "irrigation_type": None,       # e.g., "Drip irrigation"
    "recent_weather": None,        # e.g., "Continuous rain and cloudy"
    "treatments_applied": None,    # e.g., "None / Neem oil 5 days ago"
}

# ----------------- 15 REASONING INTENT CATEGORIES -----------------
INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "disease_identification": {
        "keywords": ["symptom", "blight", "spot", "mildew", "rust", "rot", "scab", "curl", "virus", "mosaic", "lesion", "identify", "what disease", "diff", "difference", "रोग", "पहचान", "लक्षण", "धब्बा", "झुलसा", "फफूंद"],
        "desc_en": "Plant disease identification and symptom verification",
        "desc_hi": "पौधे के रोग की पहचान एवं लक्षण सत्यापन"
    },
    "disease_management": {
        "keywords": ["treatment", "manage", "control", "fungicide", "cure", "spray", "prevent disease", "protect", "remedy", "उपचार", "रोकथाम", "प्रबंधन", "दवा", "कीटनाशक", "छिड़काव"],
        "desc_en": "Disease prevention, treatment and cultural management",
        "desc_hi": "रोग रोकथाम, उपचार और प्रबंधन"
    },
    "pest_management": {
        "keywords": ["insect", "pest", "caterpillar", "borer", "aphid", "whitefly", "mite", "thrip", "bug", "worm", "कीट", "कीड़ा", "इल्ली", "सफेद मक्खी", "माहू", "एफिड"],
        "desc_en": "Insect pest identification and pest management",
        "desc_hi": "कीट पहचान एवं प्रबंधन"
    },
    "soil_health": {
        "keywords": ["soil", "soil prep", "ph", "clay", "sandy", "loam", "organic matter", "humus", "salinity", "plough", "solarization", "मिट्टी", "मृदा", "जुताई", "कच्ची मिट्टी", "पीएच"],
        "desc_en": "Soil health, soil testing, pH and field preparation",
        "desc_hi": "मृदा स्वास्थ्य, परीक्षण और खेत की तैयारी"
    },
    "fertilizer_nutrition": {
        "keywords": ["fertilizer", "npk", "nitrogen", "phosphorus", "potash", "potassium", "zinc", "urea", "dap", "chlorosis", "deficiency", "micronutrient", "उर्वरक", "खाद", "यूरिया", "पोषक तत्व", "कमी"],
        "desc_en": "Fertilizers, NPK nutrition, and nutrient deficiency management",
        "desc_hi": "उर्वरक, एनपीके पोषण और पोषक तत्वों की कमी"
    },
    "irrigation_water": {
        "keywords": ["irrigation", "water", "drip", "flood", "furrow", "sprinkler", "waterlogging", "drainage", "watering", "सिंचाई", "पानी", "ड्रिप", "टपक", "जलभराव", "निकासी"],
        "desc_en": "Irrigation methods, water scheduling and moisture management",
        "desc_hi": "सिंचाई पद्धतियां, जल प्रबंधन और जलभराव"
    },
    "crop_rotation": {
        "keywords": ["rotation", "crop rotation", "intercrop", "companion", "sequence", "fallow", "legume", "solanaceous", "फसल चक्र", "क्रम", "दलहनी", "सह-फसली"],
        "desc_en": "Crop rotation, intercropping and green manuring",
        "desc_hi": "फसल चक्र, अंतःफसली और हरी खाद"
    },
    "seed_variety": {
        "keywords": ["seed", "variety", "cultivar", "hybrid", "germination", "sowing", "seed treatment", "certified seed", "बीज", "किस्म", "हाइब्रिड", "अंकुरण", "बुवाई", "बीजोपचार"],
        "desc_en": "Seed selection, hybrid varieties and seed treatments",
        "desc_hi": "बीज चयन, उन्नत किस्में और बीजोपचार"
    },
    "growth_stages": {
        "keywords": ["stage", "flowering", "vegetative", "seedling", "fruiting", "harvest time", "maturity", "अवस्था", "फूल", "वानस्पतिक", "अंकुरण", "फलन"],
        "desc_en": "Crop phenology, growth stages and seasonal timing",
        "desc_hi": "फसल की विकास अवस्थाएं और समय"
    },
    "organic_farming": {
        "keywords": ["organic", "jeevamrit", "panchgavya", "neem", "biofertilizer", "compost", "vermicompost", "natural farming", "जैविक", "जीवामृत", "नीम", "केंचुआ खाद", "प्राकृतिक खेती"],
        "desc_en": "Organic and natural farming practices",
        "desc_hi": "जैविक एवं प्राकृतिक कृषि पद्धतियां"
    },
    "cultural_practices": {
        "keywords": ["spacing", "pruning", "weeding", "mulch", "mulching", "canopy", "trellis", "staking", "निराई", "गुड़ाई", "छंटाई", "मल्चिंग", "दूरी"],
        "desc_en": "Cultural canopy management, weeding and agronomic practices",
        "desc_hi": "कृषि क्रियाएं, निराई-गुड़ाई, छंटाई और मल्चिंग"
    },
    "harvesting_postharvest": {
        "keywords": ["harvest", "picking", "storage", "post-harvest", "shelf life", "curing", "cold storage", "कटाई", "तुड़ाई", "भंडारण", "शीतगृह"],
        "desc_en": "Harvesting maturity and post-harvest handling",
        "desc_hi": "फसल कटाई और कटाई उपरांत प्रबंधन"
    },
    "general_agronomy": {
        "keywords": ["agriculture", "crop", "farming", "plant", "photosynthesis", "sunlight", "yield", "field", "खेती", "कृषि", "पैदावार", "फसल"],
        "desc_en": "General agricultural knowledge and crop biology",
        "desc_hi": "सामान्य कृषि ज्ञान और फसल जीवविज्ञान"
    },
    "government_schemes": {
        "keywords": ["scheme", "subsidy", "pm kisan", "government", "kvk", "extension", "yojana", "योजना", "सब्सिडी", "सरकारी", "केवीके"],
        "desc_en": "Government agricultural initiatives, KVK and official support",
        "desc_hi": "सरकारी कृषि योजनाएं और केवीके सहायता"
    },
    "off_topic": {
        "keywords": ["cricket", "football", "movie", "film", "python", "javascript", "code", "programming", "politics", "election", "stock market", "crypto", "bitcoin", "ipl"],
        "desc_en": "Non-agricultural / off-topic conversation",
        "desc_hi": "गैर-कृषि विषय"
    }
}

# ----------------- 17 SUPPORTED PATHOLOGY PROFILES -----------------
DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Tomato___Early_blight": {
        "symptoms_en": "Dark brown circular spots with concentric target-board rings on older lower leaves, leaf yellowing.",
        "symptoms_hi": "पुरानी निचली पत्तियों पर भूरे गोलाकार संकेंद्रित छल्लेदार धब्बे, पत्तियों का पीला पड़ना।",
        "probe_en": "Are you seeing concentric target-like rings on lower leaves, and has yellowing started?",
        "probe_hi": "क्या निचली पत्तियों पर गोल छल्लेदार धब्बे दिख रहे हैं और पत्तियां पीली पड़ रही हैं?",
        "quick_replies_en": ["Target rings on lower leaves", "Yellowing spread to middle leaves", "Spots on fruit stems", "Skip / Show Guidance"],
        "quick_replies_hi": ["निचली पत्तियों पर गोल छल्ले", "मध्यम पत्तियों पर पीलापन", "तने पर काले धब्बे", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Tomato___Late_blight": {
        "symptoms_en": "Pale green water-soaked spots rapidly turning dark brown/black; white fungal fuzz on leaf undersides in humid mornings.",
        "symptoms_hi": "पानी जैसे भीगे धब्बे जो तेजी से गहरे भूरे हो जाते हैं; सुबह नमी में पत्ती के नीचे सफेद फफूंद।",
        "probe_en": "Are leaf lesions water-soaked with white fuzz on the underside during humid mornings?",
        "probe_hi": "क्या पत्तियों पर पानी जैसे भीगे धब्बे और सुबह नीचे सफेद फफूंद दिखाई दे रही है?",
        "quick_replies_en": ["Water-soaked brown patches", "White fuzz under leaves", "Foliage collapsing fast", "Skip / Show Guidance"],
        "quick_replies_hi": ["पानी जैसे भीगे धब्बे", "पत्तियों के नीचे सफेद फफूंद", "पत्तियां तेजी से झुलस रही हैं", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms_en": "Numerous small circular spots (1-3 mm) with gray-white centers and dark brown borders on lower foliage.",
        "symptoms_hi": "असंख्य छोटे गोलाकार धब्बे (1-3 मिमी) जिनका केंद्र सफेद-भूरा और किनारा गहरा भूरा होता है।",
        "probe_en": "Are the spots small and circular with grey-white centres and dark margins on lower foliage?",
        "probe_hi": "क्या निचली पत्तियों पर भूरे-सफेद केंद्र वाले छोटे-छोटे गोल धब्बे दिख रहे हैं?",
        "quick_replies_en": ["Small spots with white centres", "Starting from bottom leaves", "Leaves drying up", "Skip / Show Guidance"],
        "quick_replies_hi": ["सफेद केंद्र वाले छोटे धब्बे", "निचली पत्तियों से शुरुआत", "पत्तियां सूख रही हैं", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Tomato___Target_Spot": {
        "symptoms_en": "Circular brown lesions on leaves and fruit with subtle concentric zones and yellow halos.",
        "symptoms_hi": "पत्तियों और फलों पर हल्के संकेंद्रित छल्लों और पीले घेरे वाले भूरे धब्बे।",
        "probe_en": "Do the spots have subtle concentric zones with a yellow halo around them?",
        "probe_hi": "क्या धब्बों के चारों ओर हल्का पीला घेरा दिखाई दे रहा है?",
        "quick_replies_en": ["Brown spots with yellow halos", "Fruit lesions visible", "Older leaves affected", "Skip / Show Guidance"],
        "quick_replies_hi": ["पीले घेरे वाले भूरे धब्बे", "फलों पर धब्बे", "पुरानी पत्तियां प्रभावित", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "symptoms_en": "Mottled light and dark green patterns on leaves, leaf distortion, blistering, and plant stunting.",
        "symptoms_hi": "पत्तियों पर हल्के और गहरे हरे रंग के चितकबरे धब्बे (मोज़ेक), पत्तियों का मुड़ना और पौधों का बौनापन।",
        "probe_en": "Are leaves showing a mottled green patchwork pattern or distortion?",
        "probe_hi": "क्या पत्तियां चितकबरी (हल्की-गहरी हरी) और मुड़ी हुई हैं?",
        "quick_replies_en": ["Mottled green mosaic pattern", "Leaves curled and distorted", "Stunted plant growth", "Skip / Show Guidance"],
        "quick_replies_hi": ["चितकबरे मोज़ेक धब्बे", "पत्तियां सिकुड़ी और मुड़ी हुई", "पौधा बौना रह गया", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms_en": "Severe upward cupping of leaves, prominent chlorotic/yellow leaf margins, bushy stunted appearance.",
        "symptoms_hi": "पत्तियों का ऊपर की ओर मुड़ना (कप की तरह), किनारों का पीला पड़ना, पौधे का झाड़ीनुमा और बौना होना।",
        "probe_en": "Are leaves curling upward like cups with yellow edges, and are whiteflies present in the crop?",
        "probe_hi": "क्या पत्तियां ऊपर की ओर मुड़ रही हैं और सफेद मक्खी दिख रही है?",
        "quick_replies_en": ["Upward leaf curling & yellowing", "Whiteflies present in field", "Flowers dropping off", "Skip / Show Guidance"],
        "quick_replies_hi": ["पत्तियां ऊपर मुड़कर पीली", "सफेद मक्खी का प्रकोप", "फूल गिर रहे हैं", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Potato___Early_blight": {
        "symptoms_en": "Dark brown to black concentric ring lesions on older potato leaves, yellowing around spots.",
        "symptoms_hi": "आलू की पुरानी पत्तियों पर गहरे भूरे या काले संकेंद्रित छल्लेदार धब्बे, आसपास पीलापन।",
        "probe_en": "Are you seeing dry target-board spots on lower potato leaves, especially after dry spells with heavy dews?",
        "probe_hi": "क्या निचली पत्तियों पर सूखे छल्लेदार धब्बे दिख रहे हैं?",
        "quick_replies_en": ["Dry concentric spots on lower leaves", "Yellowing around spots", "Spreading slowly", "Skip / Show Guidance"],
        "quick_replies_hi": ["निचली पत्तियों पर छल्लेदार धब्बे", "धब्बों के आसपास पीलापन", "फैलाव धीमा है", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Potato___Late_blight": {
        "symptoms_en": "Water-soaked lesions on leaf tips and margins turning dark brown/purplish-black; white mildew under leaves in wet weather.",
        "symptoms_hi": "पत्तियों के किनारों पर पानी से भीगे धब्बे जो तेजी से बैंगनी-काले हो जाते हैं; नम मौसम में पत्ती के नीचे सफेद फफूंद।",
        "probe_en": "Are leaf tips turning dark brown with white fuzz beneath during cold, wet, or foggy weather?",
        "probe_hi": "क्या कोहरे और ठंड में पत्तियां किनारे से काली पड़ रही हैं और नीचे सफेद फफूंद है?",
        "quick_replies_en": ["Water-soaked dark lesions", "White fuzz under leaves", "Stems rotting & smell", "Skip / Show Guidance"],
        "quick_replies_hi": ["पानी जैसे काले धब्बे", "पत्ती के नीचे सफेद फफूंद", "तने में सड़न और गंध", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Apple___Apple_scab": {
        "symptoms_en": "Olive-green to velvety dark brown/black spots on leaves and fruit, causing puckering and leaf drop.",
        "symptoms_hi": "पत्तियों और फलों पर जैतून जैसे हरे या मखमली भूरे/काले धब्बे, जिससे पत्तियां सिकुड़ कर गिरती हैं।",
        "probe_en": "Are the spots velvety olive-green to dark brown, appearing on the upper leaf surface or fruit skin?",
        "probe_hi": "क्या पत्तियों या फलों पर मखमली जैतून जैसे हरे या काले धब्बे दिख रहे हैं?",
        "quick_replies_en": ["Velvety dark spots on leaves", "Cracked/scabby fruit skin", "Early leaf drop observed", "Skip / Show Guidance"],
        "quick_replies_hi": ["पत्तियों पर मखमली काले धब्बे", "फलों की त्वचा पर खुरंड", "पत्तियां समय से पहले गिर रही हैं", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Apple___Black_rot": {
        "symptoms_en": "Frog-eye circular leaf spots with purple margins and tan centres; rotting fruit turning into hard black mummies.",
        "symptoms_hi": "पत्तियों पर बैंगनी किनारे वाले मेंढक की आंख जैसे गोल धब्बे; फल सड़कर काले और कड़े (ममी) हो जाते हैं।",
        "probe_en": "Are you seeing frog-eye spots with purple margins on leaves, or rotting shrivelled black apples?",
        "probe_hi": "क्या पत्तियों पर बैंगनी किनारे वाले गोल धब्बे या फलों का काला पड़कर सड़ना दिख रहा है?",
        "quick_replies_en": ["Frog-eye spots on leaves", "Rotting black mummified fruit", "Bark cankers on branches", "Skip / Show Guidance"],
        "quick_replies_hi": ["मेंढक की आंख जैसे धब्बे", "काले सूखे सड़े हुए फल", "टहनियों पर घाव/कैंकर", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Apple___Cedar_apple_rust": {
        "symptoms_en": "Bright yellow-orange circular spots on upper leaf surface; later tube-like fungal aecia on leaf underside.",
        "symptoms_hi": "पत्ती की ऊपरी सतह पर चमकदार पीले-नारंगी गोल धब्बे; बाद में पत्ती के नीचे बारीक नलिका जैसी संरचनाएं।",
        "probe_en": "Are the spots bright yellow-orange, and are there red cedar or juniper trees growing nearby?",
        "probe_hi": "क्या पत्तियों पर चमकदार पीले-नारंगी धब्बे हैं और क्या आसपास जुनिपर/देवदार के पेड़ हैं?",
        "quick_replies_en": ["Bright orange/yellow spots", "Underleaf tube-like structures", "Nearby cedar/juniper trees", "Skip / Show Guidance"],
        "quick_replies_hi": ["चमकदार नारंगी धब्बे", "पत्ती के नीचे नलिका जैसी संरचना", "आसपास देवदार/जुनिपर पेड़", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Corn_(maize)___Common_rust_": {
        "symptoms_en": "Golden-brown to cinnamon-brown powdery pustules scattered across both upper and lower leaf surfaces.",
        "symptoms_hi": "पत्तियों की दोनों सतहों पर सुनहरे-भूरे या दालचीनी जैसे रंग के दानेदार फफोले जो हाथ लगाने पर पाउडर छोड़ते हैं।",
        "probe_en": "Do the pustules rub off as a rusty cinnamon powder on your fingers when you touch them?",
        "probe_hi": "क्या छूने पर उंगलियों पर जंग जैसा नारंगी/भूरा पाउडर लगता है?",
        "quick_replies_en": ["Rust powder rubs off on touch", "Pustules on both leaf surfaces", "Affecting upper canopy", "Skip / Show Guidance"],
        "quick_replies_hi": ["छूने पर जंग जैसा पाउडर", "पत्ती के दोनों तरफ फफोले", "ऊपरी पत्तियां भी प्रभावित", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms_en": "Long, rectangular, tan to grayish lesions strictly delimited between leaf veins.",
        "symptoms_hi": "पत्तियों की नसों के बीच सीमित लंबे, आयताकार, भूरे या धूसर रंग के धब्बे।",
        "probe_en": "Are the lesions rectangular and running parallel between the leaf veins?",
        "probe_hi": "क्या धब्बे आयताकार हैं और पत्तियों की नसों के समानांतर चल रहे हैं?",
        "quick_replies_en": ["Rectangular vein-bound lesions", "Grayish-tan long spots", "Lower leaves first", "Skip / Show Guidance"],
        "quick_replies_hi": ["नसों के बीच आयताकार धब्बे", "लंबे भूरे धब्बे", "निचली पत्तियों से शुरुआत", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms_en": "Long elliptical cigar-shaped grayish-green to tan lesions (1-6 inches) on leaves.",
        "symptoms_hi": "पत्तियों पर बड़े सिगार के आकार के (1 से 6 इंच लंबे) अंडाकार भूरे या धूसर धब्बे।",
        "probe_en": "Are you seeing large cigar-shaped elliptical lesions across the maize foliage?",
        "probe_hi": "क्या मक्के की पत्तियों पर सिगार के आकार के बड़े लंबे धब्बे दिखाई दे रहे हैं?",
        "quick_replies_en": ["Large cigar-shaped lesions", "Grayish-green long blotches", "Spreading after wet weather", "Skip / Show Guidance"],
        "quick_replies_hi": ["सिगार जैसे बड़े लंबे धब्बे", "धूसर-हरे धब्बे", "बारिश के बाद फैलाव", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Grape___Black_rot": {
        "symptoms_en": "Small circular tan leaf spots with dark borders; shrivelling berries turning into hard black mummies.",
        "symptoms_hi": "पत्तियों पर काले किनारे वाले छोटे भूरे धब्बे; अंगूर के दाने सिकुड़कर काले और कड़े (ममी) हो जाते हैं।",
        "probe_en": "Are grapes turning brown, shrivelling into hard black mummies on the bunch?",
        "probe_hi": "क्या गुच्छे में अंगूर के दाने काले और कड़े होकर सूख रहे हैं?",
        "quick_replies_en": ["Shrivelled black berries on bunch", "Leaf spots with tiny black dots", "Affecting young shoots", "Skip / Show Guidance"],
        "quick_replies_hi": ["गुच्छों पर सूखे काले अंगूर", "पत्तियों पर बारीक काले बिंदुओं वाले धब्बे", "नई टहनियों पर प्रभाव", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms_en": "Tiger-stripe yellow/brown chlorotic zones between leaf veins; dark measles-like spots on berry skins.",
        "symptoms_hi": "पत्तियों की नसों के बीच बाघ की धारियों (टाइगर स्ट्राइप) जैसे पीले-भूरे धब्बे; अंगूर के छिलके पर खसरे जैसे काले बिंदु।",
        "probe_en": "Do the leaves show tiger-stripe patterns between veins, and do berries have tiny dark spots?",
        "probe_hi": "क्या पत्तियों पर बाघ की धारियों जैसे पैटर्न और अंगूर पर काले बिंदु हैं?",
        "quick_replies_en": ["Tiger-stripe leaf pattern", "Dark spots on berry skin", "Older vines affected", "Skip / Show Guidance"],
        "quick_replies_hi": ["बाघ की धारियों जैसा पैटर्न", "अंगूर पर काले बिंदु", "पुरानी बेलें प्रभावित", "छोड़ें / सीधा उपचार बताएं"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms_en": "Large irregular brown to dark reddish-brown blotches on mature leaves, leading to premature leaf scorch and defoliation.",
        "symptoms_hi": "परिपक्व पत्तियों पर बड़े अनियमित भूरे या लाल-भूरे धब्बे, जिससे पत्तियां झुलसकर गिर जाती हैं।",
        "probe_en": "Are there large irregular reddish-brown blotches causing mature leaves to scorch and drop?",
        "probe_hi": "क्या पुरानी पत्तियों पर बड़े लाल-भूरे धब्बे हैं और पत्तियां झुलस कर गिर रही हैं?",
        "quick_replies_en": ["Large reddish-brown leaf blotches", "Leaves scorching and dropping", "Post-monsoon spread", "Skip / Show Guidance"],
        "quick_replies_hi": ["बड़े लाल-भूरे धब्बे", "पत्तियां झुलस कर गिर रही हैं", "बारिश के बाद फैलाव", "छोड़ें / सीधा उपचार बताएं"]
    }
}

# ----------------- DIFFERENTIAL DIAGNOSTIC MATRICES -----------------
DIFFERENTIAL_PAIRS = [
    {
        "pair_id": "tomato_early_vs_septoria",
        "crop": "Tomato",
        "disease_a": "Tomato Early Blight",
        "disease_b": "Tomato Septoria Leaf Spot",
        "shared": "Both start on older lower leaves, produce brown spots, and cause foliage yellowing.",
        "distinct_a": "Early Blight produces larger (3-15mm) spots with distinct concentric target-board rings and can affect fruit.",
        "distinct_b": "Septoria produces numerous small circular spots (1-3mm) with light grey/white centres and tiny black specks (pycnidia); rarely affects fruit.",
        "key_question": "Are the spots large with concentric target rings, or small with white centres and black pinpoints?"
    },
    {
        "pair_id": "potato_early_vs_late_blight",
        "crop": "Potato",
        "disease_a": "Potato Early Blight",
        "disease_b": "Potato Late Blight",
        "shared": "Both cause dark foliar lesions, leaf death, and potential tuber damage.",
        "distinct_a": "Early Blight is a slow-spreading dry concentric ring spot occurring in warm weather with heavy dew.",
        "distinct_b": "Late Blight spreads extremely fast in cool wet/foggy conditions, creating water-soaked dark blotches with white mildew on leaf undersides and foul rot.",
        "key_question": "Is the spread rapid after wet foggy mornings with white fuzz underneath, or slow dry concentric rings?"
    },
    {
        "pair_id": "corn_rust_vs_gray_leaf_spot",
        "crop": "Corn",
        "disease_a": "Corn Common Rust",
        "disease_b": "Corn Gray Leaf Spot",
        "shared": "Both cause significant leaf lesions reducing photosynthetic area in maize.",
        "distinct_a": "Common Rust forms raised oval cinnamon-brown pustules on both upper and lower leaf surfaces that rub off powdery spores.",
        "distinct_b": "Gray Leaf Spot forms flat rectangular tan/gray lesions strictly bounded between leaf veins.",
        "key_question": "Do pustules release powdery rust on fingers, or are the lesions rectangular and vein-bounded?"
    },
    {
        "pair_id": "apple_black_rot_vs_scab",
        "crop": "Apple",
        "disease_a": "Apple Black Rot",
        "disease_b": "Apple Scab",
        "shared": "Both infect foliage and fruit, causing leaf lesions and fruit spotting.",
        "distinct_a": "Black Rot causes 'frog-eye' leaf spots with purple margins and mummified black shrivelled fruit on the tree.",
        "distinct_b": "Apple Scab causes velvety olive-green spots that turn dark brown, with corky cracked scab lesions on fruit skin.",
        "key_question": "Are leaf spots frog-eye shaped with purple margins, or velvety olive-green/brown?"
    }
]

STAGE_METADATA = {
    "crop_context": {
        "order": 1,
        "progress": 25,
        "label_en": "Understanding Your Crop",
        "label_hi": "फसल की जानकारी"
    },
    "symptom_progression": {
        "order": 2,
        "progress": 50,
        "label_en": "Analyzing Symptoms & Spread",
        "label_hi": "लक्षण व फैलाव का विश्लेषण"
    },
    "field_environment": {
        "order": 3,
        "progress": 75,
        "label_en": "Field & Environmental Conditions",
        "label_hi": "खेत की स्थिति व मौसम"
    },
    "advisory_management": {
        "order": 4,
        "progress": 100,
        "label_en": "Personalized Advisory & Management",
        "label_hi": "सटीक परामर्श व समाधान"
    }
}


class SmartFarmingAgent:
    """
    Advanced Agricultural Intelligence & Reasoning Assistant connecting
    plant pathology, general agronomy, intelligent ChromaDB RAG, and IBM Granite 4.
    """

    def __init__(self, generation_model_id: Optional[str] = None):
        self.generation_model_id = (
            generation_model_id or watsonx_service.model_id
        ).strip()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Intent Classification Engine (15 Categories + Multi-Intent)
    # -------------------------------------------------------------------------

    def classify_intents(
        self,
        user_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], str]:
        """
        Classifies incoming user questions into 15 agricultural & general categories.
        Returns a tuple of (matched_intents_list, primary_intent).
        """
        text_lower = user_text.lower()
        matched: List[str] = []

        # Check off-topic first
        off_topic_words = INTENT_DEFINITIONS["off_topic"]["keywords"]
        if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in off_topic_words):
            # Check if agricultural keywords also exist (to avoid false positives)
            agri_words = ["crop", "soil", "plant", "fertilizer", "pest", "disease", "farm", "खेती", "फसल", "पौधा"]
            if not any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in agri_words):
                return (["off_topic"], "off_topic")

        # Evaluate agricultural categories
        for intent_name, defn in INTENT_DEFINITIONS.items():
            if intent_name == "off_topic":
                continue
            keywords = defn["keywords"]
            for kw in keywords:
                # Word boundary match
                pattern = r'\b' + re.escape(kw) + r'\b' if len(kw) > 3 else r'\b' + re.escape(kw)
                if re.search(pattern, text_lower):
                    if intent_name not in matched:
                        matched.append(intent_name)
                    break

        # Fallback to general agronomy if agricultural question has no specific category
        if not matched:
            matched.append("general_agronomy")

        # Resolve primary intent
        priority_order = [
            "disease_identification", "disease_management", "pest_management",
            "fertilizer_nutrition", "soil_health", "irrigation_water",
            "crop_rotation", "seed_variety", "growth_stages", "organic_farming",
            "cultural_practices", "harvesting_postharvest", "government_schemes",
            "general_agronomy", "off_topic"
        ]
        primary = next((i for i in priority_order if i in matched), matched[0])
        return (matched, primary)

    # -------------------------------------------------------------------------
    # 2. Conversational Reference Resolution & Correction Handling
    # -------------------------------------------------------------------------

    def _resolve_references_and_corrections(
        self,
        user_text: str,
        session: Dict[str, Any],
        disease_info: Dict[str, Any]
    ) -> str:
        """
        Resolves anaphoric references ('this disease', 'that crop', 'previous spray')
        and processes explicit farmer corrections (e.g. 'Actually it is potato').
        """
        resolved_text = user_text
        text_lower = user_text.lower()

        # Handle explicit corrections
        # e.g., "Actually, it is Potato not Tomato" or "Not Tomato, it is Potato"
        m1 = re.search(r'(?:actually|in fact)?\s*,?\s*(?:it is|its|it\'s|i mean|growing|crop is)\s+([a-zA-Z\u0900-\u097F]+)\s+(?:not|instead of)\s+([a-zA-Z\u0900-\u097F]+)', text_lower)
        m2 = re.search(r'(?:not|instead of)\s+([a-zA-Z\u0900-\u097F]+)\s*,?\s*(?:it is|its|it\'s|i mean|actually)\s+([a-zA-Z\u0900-\u097F]+)', text_lower)
        if m1:
            new_crop = m1.group(1).capitalize()
            session["profile"]["crop"] = new_crop
            session["disease_context"] = None
        elif m2:
            new_crop = m2.group(2).capitalize()
            session["profile"]["crop"] = new_crop
            session["disease_context"] = None

        # Resolve "this disease"
        if re.search(r'\b(?:this|the)\s+disease\b', text_lower):
            primary_disease = None
            if disease_info.get("has_prediction"):
                primary_disease = disease_info.get("primary_disease")
            elif session.get("disease_context") and session["disease_context"].get("primary_disease"):
                primary_disease = session["disease_context"]["primary_disease"]

            if primary_disease:
                resolved_text = re.sub(r'\b(?:this|the)\s+disease\b', f"{primary_disease}", resolved_text, flags=re.IGNORECASE)

        # Resolve "this crop" / "that crop"
        if re.search(r'\b(?:this|that|my)\s+crop\b', text_lower):
            known_crop = session["profile"].get("crop")
            if known_crop:
                resolved_text = re.sub(r'\b(?:this|that)\s+crop\b', f"{known_crop}", resolved_text, flags=re.IGNORECASE)

        # Normalize common agricultural misspellings
        misspellings = {
            r'\btomto\b': "tomato",
            r'\bblite\b': "blight",
            r'\bfertilyzer\b': "fertilizer",
            r'\bfungus sprey\b': "fungicide spray",
            r'\btamatar\b': "tomato",
            r'\baaloo\b': "potato"
        }
        for bad_p, good_w in misspellings.items():
            resolved_text = re.sub(bad_p, good_w, resolved_text, flags=re.IGNORECASE)

        return resolved_text

    # -------------------------------------------------------------------------
    # 3. Differential Disease Analysis Module
    # -------------------------------------------------------------------------

    def _perform_differential_analysis(
        self,
        crop: Optional[str],
        symptoms: Optional[str],
        disease_info: Dict[str, Any],
        user_text: str,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Compares confusing diseases when symptoms overlap or user asks about differences.
        Returns a structured differential comparison or None.
        """
        is_hi = language.lower().startswith("hi")
        combined_text = f"{user_text} {symptoms or ''}".lower()
        active_crop = (crop or "").lower()

        for pair in DIFFERENTIAL_PAIRS:
            pair_crop = pair["crop"].lower()
            if active_crop and active_crop not in pair_crop and pair_crop not in active_crop:
                continue

            dis_a = pair["disease_a"].lower()
            dis_b = pair["disease_b"].lower()
            short_a = re.sub(r'^(tomato|potato|corn|apple|grape)\s+', '', dis_a)
            short_b = re.sub(r'^(tomato|potato|corn|apple|grape)\s+', '', dis_b)

            mentions_diff = any(w in combined_text for w in ["difference", "diff", "distinguish", "vs", "versus", "or", "अंतर", "तुलना"])
            has_disease_a = dis_a in combined_text or short_a in combined_text
            has_disease_b = dis_b in combined_text or short_b in combined_text
            symptom_overlap = any(s in combined_text for s in ["yellow", "spot", "blight", "rings", "fuzz", "धब्बा", "झुलसा"])

            if (mentions_diff and (has_disease_a or has_disease_b)) or (has_disease_a and has_disease_b) or (disease_info.get("has_prediction") and pair_crop in active_crop and symptom_overlap):
                return {
                    "pair_id": pair["pair_id"],
                    "crop": pair["crop"],
                    "disease_a": pair["disease_a"],
                    "disease_b": pair["disease_b"],
                    "shared_symptoms": pair["shared"] if not is_hi else "दोनों पुरानी निचली पत्तियों पर भूरे धब्बे और पीलापन पैदा करते हैं।",
                    "distinguishing_feature_a": pair["distinct_a"] if not is_hi else f"{pair['disease_a']} में बड़े संकेंद्रित छल्लेदार धब्बे होते हैं।",
                    "distinguishing_feature_b": pair["distinct_b"] if not is_hi else f"{pair['disease_b']} में छोटे गोल धब्बे सफेद केंद्र के साथ होते हैं।",
                    "targeted_probe": pair["key_question"] if not is_hi else "क्या धब्बे बड़े छल्लेदार हैं या छोटे सफेद केंद्र वाले?"
                }
        return None

    # -------------------------------------------------------------------------
    # 4. RAG Retrieval Necessity Evaluator
    # -------------------------------------------------------------------------

    def _evaluate_rag_necessity(self, intents: List[str], user_text: str) -> bool:
        """
        Determines whether document retrieval from ChromaDB is needed,
        avoiding unnecessary vector searches for off-topic or simple conversational greetings.
        """
        if "off_topic" in intents:
            return False

        simple_greetings = ["hello", "hi", "hey", "नमस्ते", "good morning", "good evening", "how are you"]
        if user_text.strip().lower() in simple_greetings:
            return False

        # RAG is required for technical agricultural queries
        rag_worthy = [
            "disease_identification", "disease_management", "pest_management",
            "fertilizer_nutrition", "soil_health", "irrigation_water",
            "crop_rotation", "seed_variety", "organic_farming", "cultural_practices"
        ]
        return any(i in rag_worthy for i in intents)

    # -------------------------------------------------------------------------
    # 5. Session Management & Progressive Profile
    # -------------------------------------------------------------------------

    def get_or_create_session(
        self,
        conversation_id: Optional[str] = None,
        crop: Optional[str] = None,
        location: Optional[str] = None,
        disease_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieves or initializes an isolated conversation session."""
        cid = conversation_id.strip() if conversation_id and conversation_id.strip() else None
        if not cid:
            cid = f"leafsense-{uuid.uuid4().hex[:12]}"

        if cid not in self._sessions:
            self._sessions[cid] = {
                "conversation_id": cid,
                "created_at": time.time(),
                "updated_at": time.time(),
                "messages": [],
                "profile": dict(DEFAULT_FARMER_PROFILE),
                "disease_context": disease_context,
                "stage": "crop_context",
                "asked_topics": set()
            }

        session = self._sessions[cid]
        session["updated_at"] = time.time()

        if crop and crop.strip() and not session["profile"].get("crop"):
            session["profile"]["crop"] = crop.strip()
        if location and location.strip() and not session["profile"].get("location"):
            session["profile"]["location"] = location.strip()
        if disease_context:
            session["disease_context"] = disease_context

        if len(self._sessions) > 1000:
            oldest_key = min(self._sessions.keys(), key=lambda k: self._sessions[k]["updated_at"])
            del self._sessions[oldest_key]

        return session

    def reset_session(self, conversation_id: str) -> Dict[str, Any]:
        """Resets session memory for a fresh conversation."""
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
        return self.get_or_create_session(conversation_id)

    def _extract_and_update_profile(
        self,
        user_text: str,
        profile: Dict[str, Any],
        scan_crop: Optional[str],
        disease_info: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Extracts farmer details from natural user utterances and updates profile state."""
        updated = dict(profile)
        text_lower = user_text.lower()

        # 1. Crop extraction
        if not updated.get("crop"):
            if scan_crop:
                updated["crop"] = scan_crop
            else:
                crop_dict = {
                    "tomato": "Tomato", "टमाटर": "Tomato",
                    "potato": "Potato", "आलू": "Potato",
                    "apple": "Apple", "सेब": "Apple",
                    "grape": "Grape", "अंगूर": "Grape",
                    "corn": "Corn (Maize)", "maize": "Corn (Maize)", "मक्का": "Corn (Maize)",
                    "wheat": "Wheat", "गेहूं": "Wheat",
                    "rice": "Rice", "धान": "Rice", "चावल": "Rice",
                    "cotton": "Cotton", "कपास": "Cotton",
                    "chilli": "Chilli", "मिर्च": "Chilli",
                    "onion": "Onion", "प्याज": "Onion",
                    "mustard": "Mustard", "सरसों": "Mustard",
                }
                for c_word, c_name in crop_dict.items():
                    if c_word in text_lower:
                        updated["crop"] = c_name
                        break

        # 2. Location extraction
        if not updated.get("location"):
            loc_match = re.search(r'\b(?:in|from|at|district|state|near|जिला|राज्य)\s+([A-Za-z\u0900-\u097F\s,]{3,30})', user_text, re.IGNORECASE)
            if loc_match:
                candidate = loc_match.group(1).strip()
                if not any(stop in candidate.lower() for stop in ["my field", "my farm", "the leaf", "this plant", "मेरे खेत", "what should", "the crop"]):
                    updated["location"] = candidate

        # 3. Plant Age extraction
        if not updated.get("plant_age"):
            age_match = re.search(r'(\d+)\s*(?:days?|weeks?|months?|दिन|सप्ताह|महीने)', text_lower)
            if age_match:
                updated["plant_age"] = age_match.group(0).strip()

        # 4. Growth stage extraction
        if not updated.get("growth_stage"):
            stage_keywords = {
                "seedling / nursery": ["seedling", "nursery", "germination", "अंकुरण", "नर्सरी", "पौध"],
                "vegetative stage": ["vegetative", "growing leaves", "canopy", "वानस्पतिक", "बढ़वार"],
                "flowering stage": ["flowering", "bloom", "blossom", "फूल", "फूल आना"],
                "fruit development": ["fruiting", "fruit development", "berry", "फल", "दाने"],
                "near harvest": ["harvest", "ripening", "mature", "कटाई", "पकाई"]
            }
            for stage_name, words in stage_keywords.items():
                if any(w in text_lower for w in words):
                    updated["growth_stage"] = stage_name
                    break

        # 5. Onset / First noticed extraction
        if not updated.get("first_noticed"):
            onset_keywords = {
                "1-2 days ago": ["yesterday", "1 day", "2 days", "just started", "कल", "एक-दो दिन"],
                "about a week ago": ["1 week", "a week", "few days", "हफ्ते भर", "कुछ दिन पहले"],
                "over 2 weeks": ["2 weeks", "many days", "long time", "दो हफ्ते", "काफी समय से"]
            }
            for onset_name, words in onset_keywords.items():
                if any(w in text_lower for w in words):
                    updated["first_noticed"] = onset_name
                    break

        # 6. Spread rate extraction
        if not updated.get("is_spreading"):
            spread_keywords = {
                "isolated (few plants)": ["few plants", "isolated", "one or two", "कुछ ही पौधों", "अलग-थलग"],
                "spreading moderately": ["spreading", "moving to other plants", "फ़ैल रहा है", "बढ़ रहा है"],
                "severe / fast spread": ["rapidly", "fast", "everywhere", "whole field", "entire plot", "तेजी से", "पूरे खेत"]
            }
            for spread_name, words in spread_keywords.items():
                if any(w in text_lower for w in words):
                    updated["is_spreading"] = spread_name
                    break

        # 7. Percentage of field affected
        if not updated.get("affected_percentage"):
            pct_match = re.search(r'(\d{1,2})\s*%', user_text)
            if pct_match:
                updated["affected_percentage"] = f"{pct_match.group(1)}%"

        # 8. Irrigation practice extraction
        if not updated.get("irrigation_type"):
            irr_keywords = {
                "drip irrigation": ["drip", "टपक", "ड्रिप"],
                "flood irrigation": ["flood", "furrow", "canal", "नहर", "बहाव"],
                "sprinkler": ["sprinkler", "फव्वारा"],
                "rainfed": ["rainfed", "dryland", "बारानी", "वर्षा आधारित"]
            }
            for irr_name, words in irr_keywords.items():
                if any(w in text_lower for w in words):
                    updated["irrigation_type"] = irr_name
                    break

        # 9. Recent weather extraction
        if not updated.get("recent_weather"):
            weather_keywords = {
                "rain & high humidity": ["rain", "raining", "humid", "cloudy", "बारिश", "नमी", "बादल"],
                "fog / cold spell": ["fog", "dew", "cold", "कोहरा", "पाला", "ठंड"],
                "dry / hot heat": ["hot", "heat", "dry", "धूप", "गर्मी", "सूखा"]
            }
            for w_name, words in weather_keywords.items():
                if any(w in text_lower for w in words):
                    updated["recent_weather"] = w_name
                    break

        # 10. Treatments applied
        if not updated.get("treatments_applied"):
            if any(w in text_lower for w in ["no spray", "no treatment", "haven't sprayed", "कोई दवा नहीं", "कुछ नहीं डाला"]):
                updated["treatments_applied"] = "None yet"
            elif any(w in text_lower for w in ["spray", "sprayed", "mancozeb", "copper", "neem", "fungicide", "छिड़काव", "दवा", "नीम"]):
                updated["treatments_applied"] = user_text[:80]

        return updated

    def _determine_stage(
        self,
        profile: Dict[str, Any],
        disease_info: Dict[str, Any],
        user_text: str
    ) -> str:
        """Determines the current conversation stage."""
        text_lower = user_text.lower()
        skip_words = ["skip", "show guidance", "tell me treatment", "cure", "spray", "how to treat", "उपचार", "दवा बताएं", "रोकथाम", "क्या करें"]
        if any(w in text_lower for w in skip_words):
            return "advisory_management"

        if not profile.get("crop") or not profile.get("location"):
            return "crop_context"

        if not profile.get("growth_stage") or not profile.get("is_spreading") or not profile.get("first_noticed"):
            return "symptom_progression"

        if not profile.get("irrigation_type") or not profile.get("recent_weather"):
            return "field_environment"

        return "advisory_management"

    def _generate_quick_replies(
        self,
        stage: str,
        profile: Dict[str, Any],
        disease_info: Dict[str, Any],
        language: str = "en"
    ) -> List[str]:
        """Generates interactive quick-response chips tailored to the current stage and disease."""
        is_hi = language.lower().startswith("hi")
        primary_raw = None
        if disease_info.get("has_prediction") and disease_info.get("diseases"):
            primary_raw = disease_info["diseases"][0].get("raw_label")

        if primary_raw and primary_raw in DISEASE_PROFILES:
            d_spec = DISEASE_PROFILES[primary_raw]
            return d_spec["quick_replies_hi"] if is_hi else d_spec["quick_replies_en"]

        if stage == "crop_context":
            if is_hi:
                return ["वानस्पतिक बढ़वार", "फूल आने की अवस्था", "फल बनने की अवस्था", "छोड़ें / सीधा उपचार बताएं"]
            return ["Vegetative stage", "Flowering stage", "Fruit development", "Skip / Show Guidance"]

        elif stage == "symptom_progression":
            if is_hi:
                return ["1-2 दिन पहले दिखा", "हफ्ते भर से है", "तेजी से फैल रहा है", "छोड़ें / सीधा उपचार बताएं"]
            return ["Started 1-2 days ago", "About a week ago", "Spreading to other plants", "Skip / Show Guidance"]

        elif stage == "field_environment":
            if is_hi:
                return ["ड्रिप (टपक) सिंचाई", "नहर/बहाव सिंचाई", "हाल ही में बारिश हुई", "छोड़ें / सीधा उपचार बताएं"]
            return ["Drip irrigation", "Flood irrigation", "Recent continuous rain", "Skip / Show Guidance"]

        else:
            if is_hi:
                return ["जैविक उपचार बताएं", "रोकथाम के उपाय", "KVK से संपर्क कैसे करें?", "नया परामर्श शुरू करें"]
            return ["Organic remedies", "Preventive management", "How to contact local KVK?", "Start new consultation"]

    # -------------------------------------------------------------------------
    # Legacy Context Helpers
    # -------------------------------------------------------------------------

    def _infer_crop_from_diseases(self, diseases: List[str]) -> Optional[str]:
        """Infers the plant/crop name from raw disease labels."""
        if not diseases:
            return None
        first_disease = diseases[0]
        if "___" in first_disease:
            crop_part = first_disease.split("___")[0].replace("_", " ").strip()
            if "(" in crop_part:
                crop_part = crop_part.split("(")[0].strip()
            return crop_part
        return None

    def _format_disease_context(
        self,
        disease_predictions: Optional[Dict[str, float]],
        detected_diseases: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Structures disease prediction metrics, formatted names, and confidence bands."""
        predictions_map = disease_predictions or {}
        detected_list = detected_diseases or list(predictions_map.keys())

        items = []
        max_confidence = 0.0

        for disease in detected_list:
            conf = predictions_map.get(disease, 0.0)
            if conf > max_confidence:
                max_confidence = conf

            items.append({
                "raw_label": disease,
                "formatted_name": format_disease_name(disease),
                "confidence": round(float(conf), 4),
                "percentage": f"{round(float(conf) * 100, 1)}%" if conf > 0 else "Detected"
            })

        return {
            "has_prediction": len(items) > 0,
            "diseases": items,
            "max_confidence": max_confidence,
            "primary_disease": items[0]["formatted_name"] if items else None
        }

    def _build_retrieval_query(
        self,
        question: str,
        crop: Optional[str],
        disease_info: Dict[str, Any]
    ) -> str:
        """Synthesizes a high-signal search query to pull pertinent chunks from ChromaDB."""
        tokens: List[str] = []
        if crop and crop.strip():
            tokens.append(crop.strip())
        if disease_info.get("has_prediction") and disease_info.get("primary_disease"):
            tokens.append(disease_info["primary_disease"])
        tokens.append(question.strip())
        return " ".join(tokens)

    def _assess_confidence(
        self,
        disease_info: Dict[str, Any],
        grounded: bool,
        language: str = "en"
    ) -> str:
        """Explains AI certainty and limitations clearly to prevent misplaced trust in visual models."""
        is_hi = language.lower().startswith("hi")

        if disease_info.get("has_prediction"):
            max_conf = disease_info["max_confidence"]
            primary = disease_info["primary_disease"]

            if max_conf >= 0.80:
                conf_level = "High visual similarity" if not is_hi else "उच्च विजुअल समानता"
                advice = (
                    f"Model indicates {primary} ({round(max_conf * 100, 1)}% probability). "
                    f"Visual symptoms strongly align, but ground confirmation of leaf texture and stem health is advised."
                    if not is_hi else
                    f"मॉडल {primary} ({round(max_conf * 100, 1)}% संभावना) का संकेत देता है। "
                    f"लक्षण स्पष्ट हैं, लेकिन पत्ती और तने के प्रत्यक्ष निरीक्षण की सिफारिश की जाती है।"
                )
            elif max_conf >= 0.50:
                conf_level = "Moderate visual similarity" if not is_hi else "मध्यम विजुअल समानता"
                advice = (
                    f"Model detects {primary} ({round(max_conf * 100, 1)}% probability). "
                    f"Visual patterns may overlap with other nutrient deficiencies or leaf spots. Field inspection is recommended."
                    if not is_hi else
                    f"मॉडल {primary} ({round(max_conf * 100, 1)}% संभावना) का संकेत देता है। "
                    f"पोषक तत्वों की कमी या अन्य धब्बों से समानता हो सकती है। विशेषज्ञ जांच की सिफारिश की जाती है।"
                )
            else:
                conf_level = "Low / Uncertain" if not is_hi else "कम / अनिश्चित"
                advice = (
                    f"Model prediction confidence is low. Leaf symptoms are ambiguous. "
                    f"Please consult your local Krishi Vigyan Kendra (KVK) for direct diagnosis."
                    if not is_hi else
                    f"मॉडल भविष्यवाणी की निश्चितता कम है। कृपया सटीक पहचान के लिए अपने नजदीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें।"
                )

            evidence_note = (
                "Verified agricultural documentation was retrieved to ground this advice."
                if grounded else
                "No verified reference documents were found in the knowledge base; recommendations are conservative general guidance."
            ) if not is_hi else (
                "सलाह को सत्यापित करने के लिए कृषि संदर्भ दस्तावेज प्राप्त किए गए हैं।"
                if grounded else
                "ज्ञानकोश में कोई सत्यापित संदर्भ दस्तावेज नहीं मिला; सलाह सामान्य कृषि सिद्धांतों पर आधारित है।"
            )

            return f"[{conf_level}] {advice} {evidence_note}"
        else:
            if grounded:
                return (
                    "Advice is grounded in verified agricultural literature from the knowledge base."
                    if not is_hi else
                    "यह सलाह ज्ञानकोश में उपलब्ध सत्यापित कृषि साहित्य पर आधारित है।"
                )
            else:
                return (
                    "No image prediction or verified literature was provided. Recommendations are general agronomic guidance."
                    if not is_hi else
                    "कोई छवि भविष्यवाणी या सत्यापित दस्तावेज उपलब्ध नहीं है। यह सामान्य कृषि सलाह है।"
                )

    def _generate_follow_up_questions(
        self,
        crop: Optional[str],
        location: Optional[str],
        disease_info: Dict[str, Any],
        language: str = "en"
    ) -> List[str]:
        """Generates intuitive follow-up questions to elicit missing situational details from the farmer."""
        is_hi = language.lower().startswith("hi")
        questions: List[str] = []

        if not location or not location.strip():
            questions.append(
                "What is your district or state? (Weather and disease patterns differ across agro-climatic zones)."
                if not is_hi else
                "आपका जिला या राज्य कौन सा है? (विभिन्न कृषि-जलवायु क्षेत्रों में मौसम और रोग का प्रभाव अलग होता है)।"
            )

        if not crop or not crop.strip():
            questions.append(
                "Which crop or variety are you growing?"
                if not is_hi else
                "आप किस फसल या किस्म की खेती कर रहे हैं?"
            )

        questions.append(
            "What is the current growth stage of your crop (e.g., seedling, vegetative, flowering, or fruit development)?"
            if not is_hi else
            "आपकी फसल वर्तमान में किस अवस्था में है (जैसे अंकुरण, वानस्पतिक, फूल आने की, या फल बनने की अवस्था)?"
        )

        if disease_info.get("has_prediction"):
            questions.append(
                "Are symptoms localized to a few isolated plants or spread across the entire plot?"
                if not is_hi else
                "क्या रोग के लक्षण कुछ पौधों तक ही सीमित हैं या पूरे खेत में फैल रहे हैं?"
            )

        return questions[:3]

    # -------------------------------------------------------------------------
    # 6. Safety & Response Validation Guardrail
    # -------------------------------------------------------------------------

    def _validate_response_safety(
        self,
        answer_text: str,
        is_grounded: bool,
        intents: List[str],
        language: str = "en"
    ) -> str:
        """
        Enforces responsible AI principles:
        - Prevents unverified toxic chemical cocktail dosage hallucination.
        - Appends clear extension notice when chemical advice is requested without RAG literature.
        """
        is_hi = language.lower().startswith("hi")
        cleaned = answer_text

        # If user asked for chemical dosage without verified RAG backing
        if ("disease_management" in intents or "pest_management" in intents) and not is_grounded:
            if re.search(r'\b(?:ml|grams?|gm|litres?|spray\s+\d+)\b', cleaned, re.IGNORECASE):
                disclaimer_addition = (
                    "\n\n⚠️ **Chemical Safety Notice:** Exact pesticide dosage depends on local CIB&RC registration, "
                    "formulation concentration, and sprayer calibration. Please confirm exact chemical dosage with "
                    "your local Krishi Vigyan Kendra (KVK) or agricultural extension officer before spraying."
                    if not is_hi else
                    "\n\n⚠️ **रासायनिक सुरक्षा सूचना:** रासायनिक कीटनाशक की सही मात्रा CIB&RC पंजीकरण, उत्पाद लेबल और "
                    "स्प्रेयर कैलिब्रेशन पर निर्भर करती है। किसी भी रसायन का छिड़काव करने से पहले अपने नजदीकी "
                    "कृषि विज्ञान केंद्र (KVK) से सही मात्रा की पुष्टि अवश्य करें।"
                )
                cleaned += disclaimer_addition

        return cleaned

    # -------------------------------------------------------------------------
    # 7. Main Advanced Reasoning Flow
    # -------------------------------------------------------------------------

    def get_recommendation(
        self,
        question: str,
        crop: Optional[str] = None,
        location: Optional[str] = None,
        disease_predictions: Optional[Dict[str, float]] = None,
        detected_diseases: Optional[List[str]] = None,
        language: str = "en",
        top_k: int = 4,
        conversation_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        farmer_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full 10-step Advanced Reasoning Workflow:
        1. Context & reference resolution.
        2. Multi-intent classification.
        3. Session memory & progressive profile extraction.
        4. Differential disease analysis evaluation.
        5. RAG necessity assessment & retrieval.
        6. IBM Granite 4 prompt formulation.
        7. Model inference.
        8. Safety validation & guardrails.
        9. Reasoning summary generation.
        10. Return rich structured response.
        """
        if not question or not question.strip():
            raise ValueError("Farmer question cannot be empty.")

        lang = (language or "en").lower().strip()
        is_hi = lang.startswith("hi")

        # 1. Disease Context Formatting
        disease_info = self._format_disease_context(disease_predictions, detected_diseases)

        # 2. Crop Resolution
        resolved_crop = crop.strip() if crop and crop.strip() else None
        if not resolved_crop and disease_info["has_prediction"]:
            raw_labels = [d["raw_label"] for d in disease_info["diseases"]]
            resolved_crop = self._infer_crop_from_diseases(raw_labels)

        # 3. Retrieve or Create Session Memory
        session = self.get_or_create_session(
            conversation_id=conversation_id,
            crop=resolved_crop,
            location=location,
            disease_context=disease_info if disease_info["has_prediction"] else None
        )
        active_cid = session["conversation_id"]

        # Merge client-provided farmer profile if sent
        if farmer_profile and isinstance(farmer_profile, dict):
            for k, v in farmer_profile.items():
                if v and not session["profile"].get(k):
                    session["profile"][k] = v

        # Ingest external message history if provided
        if messages and isinstance(messages, list):
            session["messages"] = [
                {"role": m.get("role", "user"), "content": m.get("content", m.get("text", ""))}
                for m in messages[-6:]
                if m.get("content") or m.get("text")
            ]

        # 4. Resolve References & Corrections
        resolved_question = self._resolve_references_and_corrections(
            user_text=question,
            session=session,
            disease_info=disease_info
        )

        # 5. Intent Classification (15 Categories + Multi-Intent)
        intents, primary_intent = self.classify_intents(resolved_question, session)

        # Handle purely off-topic queries immediately without external API cost
        if primary_intent == "off_topic":
            off_topic_msg = (
                "I am LeafSense, specialized exclusively in agriculture, crop health, and smart farming advisory. "
                "I can assist you with plant disease diagnosis, soil preparation, fertilizers, irrigation, pest management, "
                "and crop cultivation practices. How can I help you with your crops or farm today?"
                if not is_hi else
                "मैं लीफसेंस (LeafSense) हूँ, जो विशेष रूप से कृषि, फसल स्वास्थ्य और आधुनिक कृषि परामर्श के लिए समर्पित है। "
                "मैं फसल रोगों की पहचान, मृदा तैयारी, उर्वरक प्रबंधन, सिंचाई और कीट नियंत्रण में आपकी सहायता कर सकता हूँ। "
                "आज मैं आपकी फसल या खेत के संबंध में क्या जानकारी दे सकता हूँ?"
            )
            return {
                "success": True,
                "conversation_id": active_cid,
                "question": question,
                "answer": off_topic_msg,
                "intents": intents,
                "primary_intent": primary_intent,
                "crop": session["profile"].get("crop") or resolved_crop,
                "location": session["profile"].get("location") or location,
                "disease_context": None,
                "grounded": False,
                "evidence_type": "off_topic_redirection",
                "sources": [],
                "confidence_assessment": "Question is outside the scope of agricultural advisory.",
                "follow_up_questions": [
                    "Ask about plant disease detection" if not is_hi else "पौधों के रोगों के बारे में पूछें",
                    "Ask about soil and fertilizer management" if not is_hi else "मिट्टी और खाद प्रबंधन के बारे में पूछें",
                    "Ask about drip irrigation best practices" if not is_hi else "ड्रिप सिंचाई के बारे में पूछें"
                ],
                "quick_replies": [
                    "Check crop diseases" if not is_hi else "फसल रोग की जांच करें",
                    "Soil preparation tips" if not is_hi else "खेत की तैयारी के उपाय",
                    "Fertilizer scheduling" if not is_hi else "उर्वरक की सही मात्रा"
                ],
                "stage": "crop_context",
                "stage_label": STAGE_METADATA["crop_context"]["label_hi"] if is_hi else STAGE_METADATA["crop_context"]["label_en"],
                "farmer_profile": session["profile"],
                "disclaimer": STANDARD_DISCLAIMER_HI if is_hi else STANDARD_DISCLAIMER_EN,
                "language": lang,
                "model_used": self.generation_model_id,
                "reasoning_summary": "Recognized non-agricultural query; politely redirected to farming domains."
            }

        # 6. Extract and Update Progressive Profile
        session["profile"] = self._extract_and_update_profile(
            user_text=resolved_question,
            profile=session["profile"],
            scan_crop=resolved_crop,
            disease_info=disease_info,
            language=lang
        )
        current_crop = session["profile"].get("crop") or resolved_crop
        current_location = session["profile"].get("location") or location

        # 7. Determine Consultation Stage
        active_stage = self._determine_stage(session["profile"], disease_info, resolved_question)
        session["stage"] = active_stage
        stage_meta = STAGE_METADATA.get(active_stage, STAGE_METADATA["crop_context"])
        stage_label = stage_meta["label_hi"] if is_hi else stage_meta["label_en"]

        # 8. Differential Disease Analysis
        differential_analysis = self._perform_differential_analysis(
            crop=current_crop,
            symptoms=session["profile"].get("symptoms"),
            disease_info=disease_info,
            user_text=resolved_question,
            language=lang
        )

        # 9. Dynamic RAG Retrieval & Intelligent Ranking
        needs_rag = self._evaluate_rag_necessity(intents, resolved_question)
        retrieved_chunks: List[Dict[str, Any]] = []

        if needs_rag:
            retrieval_query = self._build_retrieval_query(resolved_question, current_crop, disease_info)
            retrieved_chunks = rag_service.retrieve_context(
                query=retrieval_query,
                top_k=top_k,
                similarity_threshold=0.20
            )

        is_grounded = len(retrieved_chunks) > 0
        evidence_type = "verified_rag" if is_grounded else ("general_knowledge" if not disease_info.get("has_prediction") else "uncertain_preliminary")

        # 10. Construct Structured System Prompt for IBM Granite 4
        target_lang = "Hindi (हिंदी)" if is_hi else "simple, farmer-friendly English"

        system_instruction = (
            f"You are the LeafSense Agricultural Intelligence Assistant, an advanced agronomy reasoning consultant "
            f"supporting Indian farmers.\n\n"
            f"REASONING & COMMUNICATION WORKFLOW:\n"
            f"1. Understand the farmer's core inquiry across detected intents ({', '.join(intents)}).\n"
            f"2. Structure your response adaptively:\n"
            f"   - Direct Answer: Address the question warmly and clearly in simple language.\n"
            f"   - Agricultural Concept: Explain technical terms simply (e.g., explain 'concentric rings', 'NPK ratio', 'chlorosis', or 'soil solarization').\n"
            f"   - Evidence & Rationale: Cite retrieved literature when available. Clearly state if advice is based on general agronomic principles.\n"
            f"   - Practical Actionable Steps: Provide 2-4 safe, concrete cultural, organic, or management steps.\n"
            f"   - Safety & Precaution: Remind the farmer of official CIB&RC guidelines or KVK consultation before chemical application.\n"
            f"3. Strict Non-Hallucination: NEVER invent specific pesticide dosages, unverified chemical cocktails, spray frequencies, or mandi prices.\n"
            f"4. Preliminary Vision Disclaimer: Disease scan predictions are preliminary visual indicators, never confirmed laboratory diagnoses.\n"
            f"5. Conversational Follow-up: Ask ONE targeted follow-up question only if vital information is missing. If the farmer asked for direct guidance or already answered, do not force redundant questions.\n"
            f"6. Language: Fluently generate in {target_lang}."
        )

        # Build context prompt
        prompt_sections: List[str] = []

        # Known Farmer Profile block
        prompt_sections.append("KNOWN FARMER & FIELD PROFILE:")
        for k, v in session["profile"].items():
            if v:
                label_clean = k.replace("_", " ").title()
                prompt_sections.append(f"• {label_clean}: {v}")

        # Active Scan context
        if disease_info.get("has_prediction"):
            diag_str = ", ".join([f"{d['formatted_name']} ({d['percentage']})" for d in disease_info["diseases"]])
            conf_val = disease_info["max_confidence"]
            uncertainty_note = " (PRELIMINARY SCAN: Confidence < 50%, communicate visual ambiguity)" if conf_val < 0.50 else " (PRELIMINARY SCAN)"
            prompt_sections.append(f"• Image Model Preliminary Prediction: {diag_str}{uncertainty_note}")
        else:
            prompt_sections.append("• Image Model Preliminary Prediction: None (General agricultural inquiry)")

        # Differential Diagnosis Context
        if differential_analysis:
            prompt_sections.append("\nDIFFERENTIAL PATHOLOGY COMPARISON:")
            prompt_sections.append(f"• Comparing {differential_analysis['disease_a']} vs {differential_analysis['disease_b']}")
            prompt_sections.append(f"• Shared Symptoms: {differential_analysis['shared_symptoms']}")
            prompt_sections.append(f"• Distinguishing Markers for {differential_analysis['disease_a']}: {differential_analysis['distinguishing_feature_a']}")
            prompt_sections.append(f"• Distinguishing Markers for {differential_analysis['disease_b']}: {differential_analysis['distinguishing_feature_b']}")

        # Verified Knowledge Base block
        prompt_sections.append("\nVERIFIED AGRICULTURAL REFERENCE EXCERPTS:")
        if is_grounded:
            for idx, c in enumerate(retrieved_chunks, 1):
                auth_tag = f" [{c.get('authority_type', 'reference').upper()}]"
                prompt_sections.append(f"--- [Source {idx}: {c['document']} (Page {c['page']}){auth_tag}] ---\n{c['snippet']}")
        else:
            prompt_sections.append("[Notice: No matching documents found in the LeafSense knowledge base for this query.]")

        # Recent conversation turns
        if session["messages"]:
            prompt_sections.append("\nRECENT CONVERSATION TURNS:")
            for m in session["messages"][-4:]:
                sender = "Farmer" if m.get("role") in ("user", "farmer") else "Assistant"
                prompt_sections.append(f"{sender}: {m.get('content', '')}")

        # Active query
        prompt_sections.append(f"\nCURRENT FARMER QUESTION:\n{resolved_question.strip()}")
        prompt_sections.append(
            "\nPlease formulate an adaptive, empathetic, and evidence-grounded response following the reasoning workflow."
        )

        full_prompt = "\n".join(prompt_sections)

        # 11. Generate Reasoned Response using IBM Granite 4
        try:
            generated_text = watsonx_service.generate_text(
                prompt=full_prompt,
                system_prompt=system_instruction,
                max_new_tokens=650,
                temperature=0.32,
                decoding_method="greedy",
                model_id=self.generation_model_id
            )
        except Exception as e:
            logger.error(f"Error during Granite generation in agent service: {e}")
            raise WatsonxServiceError(
                f"Agent inference failed: {e}",
                category="AGENT_INFERENCE_FAILED"
            )

        # 12. Safety & Guardrail Validation
        validated_text = self._validate_response_safety(
            answer_text=generated_text,
            is_grounded=is_grounded,
            intents=intents,
            language=lang
        )

        # 13. Update Session Message History
        session["messages"].append({"role": "user", "content": question.strip()})
        session["messages"].append({"role": "assistant", "content": validated_text.strip()})

        # 14. Format Sources Summary
        sources_summary = [
            {
                "document": c["document"],
                "page": c["page"],
                "relevance_score": c["relevance_score"],
                "authority_type": c.get("authority_type", "agricultural_reference"),
                "snippet": c["snippet"][:200] + "..." if len(c["snippet"]) > 200 else c["snippet"]
            }
            for c in retrieved_chunks
        ]

        # 15. User-Visible Reasoning Summary
        intents_desc_en = ", ".join([INTENT_DEFINITIONS.get(i, {}).get("desc_en", i) for i in intents[:2]])
        intents_desc_hi = ", ".join([INTENT_DEFINITIONS.get(i, {}).get("desc_hi", i) for i in intents[:2]])

        if differential_analysis:
            dis_a = differential_analysis["disease_a"]
            dis_b = differential_analysis["disease_b"]
            diff_text_en = f"Conducted differential comparison for {dis_a} vs {dis_b}."
            diff_text_hi = f"{dis_a} और {dis_b} के बीच विभेदक तुलना की गई।"
        else:
            diff_text_en = "Evaluated field symptoms and farmer profile context."
            diff_text_hi = "खेत के लक्षणों और किसान विवरण का विश्लेषण किया गया।"

        if is_grounded:
            rag_text_en = f"Retrieved {len(retrieved_chunks)} verified agricultural excerpts from knowledge base."
            rag_text_hi = f"कृषि ज्ञानकोश से {len(retrieved_chunks)} सत्यापित दस्तावेज प्राप्त किए गए।"
        else:
            rag_text_en = "Knowledge base contains no matching documents; evaluated general agronomic principles."
            rag_text_hi = "ज्ञानकोश में कोई सत्यापित दस्तावेज नहीं मिला; सामान्य कृषि सिद्धांतों के आधार पर विश्लेषण किया गया।"

        if not is_hi:
            reasoning_summary = f"Classified inquiry under: {intents_desc_en}. {rag_text_en} {diff_text_en}"
        else:
            reasoning_summary = f"पूछे गए विषय: {intents_desc_hi}। {rag_text_hi} {diff_text_hi}"

        confidence_assessment = self._assess_confidence(disease_info, is_grounded, language=lang)
        follow_up_questions = self._generate_follow_up_questions(current_crop, current_location, disease_info, language=lang)
        quick_replies = self._generate_quick_replies(active_stage, session["profile"], disease_info, language=lang)
        disclaimer = STANDARD_DISCLAIMER_HI if is_hi else STANDARD_DISCLAIMER_EN

        return {
            "success": True,
            "conversation_id": active_cid,
            "question": question,
            "answer": validated_text,
            "intents": intents,
            "primary_intent": primary_intent,
            "crop": current_crop,
            "location": current_location,
            "disease_context": disease_info if disease_info["has_prediction"] else None,
            "differential_analysis": differential_analysis,
            "grounded": is_grounded,
            "evidence_type": evidence_type,
            "sources": sources_summary,
            "confidence_assessment": confidence_assessment,
            "follow_up_questions": follow_up_questions,
            "quick_replies": quick_replies,
            "stage": active_stage,
            "stage_label": stage_label,
            "farmer_profile": session["profile"],
            "disclaimer": disclaimer,
            "language": lang,
            "model_used": self.generation_model_id,
            "reasoning_summary": reasoning_summary
        }


# Singleton instance for application reuse
agent_service = SmartFarmingAgent()
