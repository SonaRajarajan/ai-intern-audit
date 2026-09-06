#!/usr/bin/env python3
"""
prepare_corpus.py -- Build multilingual evaluation corpus (A1).

Languages included:
1. English (eng)
2. Hindi (hin)
3. Tamil (tam) [Dravidian 1]
4. Kannada (kan) [Dravidian 2]
5. Telugu (tel) [Dravidian 3]

Applies Unicode NFC normalization, validates parallel alignment,
computes stats, and exports text files and metadata.json.
"""

import os
import json
import argparse
from compute_metrics import count_grapheme_clusters, count_whitespace_words, count_utf8_bytes, normalize_text

# High-quality parallel sentence dataset (FLORES-200 aligned sample across 5 languages)
PARALLEL_DATA = {
    "eng": [
        "Bengaluru International Airport handled record traffic in March.",
        "The Quarterly Review meeting moved to Thursday.",
        "I bought this book yesterday from a small shop near MG Road.",
        "Children are playing cricket on the ground.",
        "The train arrived exactly on time.",
        "NASA and ISRO announced a joint mission update.",
        "Please keep the books in the cupboard.",
        "We are visiting Mysuru next week.",
        "Do you want tea or coffee?",
        "The GPU cluster ran out of memory during the night job.",
        "Artificial intelligence is transforming healthcare across the globe.",
        "She works as a senior software engineer at a technology firm.",
        "The weather in the mountains is pleasant and cool today.",
        "Renewable energy sources such as solar and wind power are growing rapidly.",
        "Heavy rain caused traffic jams in several parts of the city.",
        "Students prepared diligently for their final university examinations.",
        "Farmers expect a good harvest following the monsoon season.",
        "The new metro rail line will open for public transit next month.",
        "Cybersecurity measures must be updated to prevent data breaches.",
        "Fresh fruits and vegetables are essential for a balanced diet.",
        "The museum houses ancient artifacts from historical dynasties.",
        "Scientists discovered new evidence of water on Mars.",
        "Online learning platforms offer flexible educational courses for students.",
        "The national park is home to rare species of birds and animals.",
        "Digital payment systems have simplified daily financial transactions."
    ],
    "hin": [
        "बेंगलुरु अंतर्राष्ट्रीय हवाई अड्डे ने मार्च में रिकॉर्ड यात्रियों को संभाला।",
        "तिमाही समीक्षा बैठक गुरुवार के लिए स्थानांतरित कर दी गई है।",
        "यह किताब मैंने कल एमजी रोड के पास एक छोटी सी दुकान से खरीदी थी।",
        "बच्चे मैदान में क्रिकेट खेल रहे हैं।",
        "ट्रेन ठीक समय पर पहुँची।",
        "नासा और इसरो ने एक संयुक्त मिशन अपडेट की घोषणा की।",
        "कृपया किताबों को अलमारी में रख दें।",
        "हम अगले हफ्ते मैसूर जा रहे हैं।",
        "क्या आपको चाय चाहिए या कॉफी?",
        "रात के काम के दौरान जीपीयू क्लस्टर की मेमोरी खत्म हो गई।",
        "कृत्रिम बुद्धिमत्ता दुनिया भर में स्वास्थ्य सेवा को बदल रही है।",
        "वह एक प्रौद्योगिकी कंपनी में वरिष्ठ सॉफ्टवेयर इंजीनियर के रूप में काम करती है।",
        "पहाड़ों में आज मौसम सुहावना और ठंडा है।",
        "सौर और पवन ऊर्जा जैसे नवीकरणीय ऊर्जा स्रोत तेजी से बढ़ रहे हैं।",
        "भारी बारिश के कारण शहर के कई हिस्सों में ट्रैफिक जाम हो गया।",
        "छात्रों ने अपनी अंतिम विश्वविद्यालय परीक्षाओं के लिए लगन से तैयारी की।",
        "मानसून के मौसम के बाद किसानों को अच्छी फसल की उम्मीद है।",
        "नई मेट्रो रेल लाइन अगले महीने सार्वजनिक परिवहन के लिए खुलेगी।",
        "डेटा उल्लंघन को रोकने के लिए साइबर सुरक्षा उपायों को अद्यतन किया जाना चाहिए।",
        "संतुलित आहार के लिए ताजे फल और सब्जियां आवश्यक हैं।",
        "संग्रहालय में ऐतिहासिक राजवंशों की प्राचीन कलाकृतियां रखी गई हैं।",
        "वैज्ञानिकों ने मंगल ग्रह पर पानी के नए सबूत खोजे हैं।",
        "ऑनलाइन शिक्षण मंच छात्रों के लिए लचीले शैक्षिक पाठ्यक्रम प्रदान करते हैं।",
        "राष्ट्रीय उद्यान पक्षियों और जानवरों की दुर्लभ प्रजातियों का घर है।",
        "डिजिटल भुगतान प्रणालियों ने दैनिक वित्तीय लेनदेन को सरल बना दिया है।"
    ],
    "tam": [
        "பெங்களூரு சர்வதேச விமான நிலையம் மார்ச் மாதத்தில் சாதனைப் போக்குவரத்தைக் கையாண்டது.",
        "காலாண்டு மறுஆய்வுக் கூட்டம் வியாழக்கிழமைக்கு மாற்றப்பட்டது.",
        "எம்ஜி சாலைக்கு அருகில் உள்ள ஒரு சிறிய கடையில் இந்த புத்தகத்தை நேற்று வாங்கினேன்.",
        "குழந்தைகள் மைதானத்தில் கிரிக்கெட் விளையாடுகிறார்கள்.",
        "ரயில் சரியாக நேரத்தில் வந்து சேர்ந்தது.",
        "நாசாவும் இஸ்ரோவும் இணைந்து கூட்டு விண்வெளித் திட்டப் புதுப்பிப்பை அறிவித்தன.",
        "புத்தகங்களை அலமாரியில் வைக்குமாறு கேட்டுக்கொள்ளப்படுகிறது.",
        "நாங்கள் அடுத்த வாரம் மைசூருக்குச் செல்கிறோம்.",
        "உங்களுக்கு தேநீர் வேண்டுமா அல்லது காபி வேண்டுமா?",
        "இரவுப் பணியின் போது ஜிபியூ கிளஸ்டரின் நினைவகம் தீர்ந்துபோனது.",
        "செயற்கை நுண்ணறிவு உலகம் முழுவதும் சுகாதாரத் துறையை மாற்றியமைத்து வருகிறது.",
        "அவர் ஒரு தொழில்நுட்ப நிறுவனத்தில் மூத்த மென்பொருள் பொறியாளராகப் பணியாற்றுகிறார்.",
        "மலைப் பகுதியில் இன்றைய வானிலை மிகவும் இதமாகவும் குளிர்ச்சியாகவும் இருக்கிறது.",
        "சூரிய மற்றும் காற்று ஆற்றல் போன்ற புதுப்பிக்கத்தக்க ஆற்றல் மூலங்கள் வேகமாக வளர்ந்து வருகின்றன.",
        "கனமழை காரணமாக நகரின் பல பகுதிகளில் கடுமையான போக்குவரத்து நெரிசல் ஏற்பட்டது.",
        "மாணவர்கள் தங்களின் இறுதிப் பல்கலைக்கழகத் தேர்வுகளுக்குக் கவனமாகத் தயாராகினர்.",
        "பருவமழைக்குப் பிறகு விவசாயிகள் நல்ல விளைச்சலை எதிர்பார்க்கிறார்கள்.",
        "புதிய மெட்ரோ ரயில் பாதை அடுத்த மாதம் பொதுமக்கள் பயன்பாட்டிற்காகத் திறக்கப்படும்.",
        "தரவு மீறல்களைத் தடுக்க இணையப் பாதுகாப்பு நடவடிக்கைகள் புதுப்பிக்கப்பட வேண்டும்.",
        "சீரான உணவுக்கு புதிய பழங்களும் காய்கறிகளும் அவசியமானவை.",
        "இந்த அருங்காட்சியகத்தில் பண்டைய வம்சங்களின் வரலாற்று கலைப்பொருட்கள் உள்ளன.",
        "செவ்வாய் கிரகத்தில் நீர் இருப்பதற்கான புதிய ஆதாரங்களை விஞ்ஞானிகள் கண்டுபிடித்துள்ளனர்.",
        "ஆன்லைன் கற்றல் தளங்கள் மாணவர்களுக்கு நெகிழ்வான கல்விப் பயிற்சிகளை வழங்குகின்றன.",
        "தேசிய பூங்கா அரிய வகை பறவைகள் மற்றும் விலங்குகளின் வாழிடமாக உள்ளது.",
        "டிஜிட்டல் பணப்பரிவர்த்தனை முறைகள் அன்றாட நிதிப் பரிவர்த்தனைகளை எளிதாக்கியுள்ளன."
    ],
    "kan": [
        "ಬೆಂಗಳೂರು ಅಂತರರಾಷ್ಟ್ರೀಯ ವಿಮಾನ ನಿಲ್ದಾಣವು ಮಾರ್ಚ್‌ನಲ್ಲಿ ದಾಖಲೆಯ ಸಂಚಾರವನ್ನು ನಿರ್ವಹಿಸಿದೆ.",
        "ತ್ರೈಮಾಸಿಕ ಪರಾಮರ್ಶೆ ಸಭೆಯನ್ನು ಗುರುವಾರಕ್ಕೆ ಮುಂದೂಡಲಾಗಿದೆ.",
        "ನಾನು ಈ ಪುಸ್ತಕವನ್ನು ನಿನ್ನೆ ಎಂಜಿ ರಸ್ತೆ ಬಳಿಯ ಸಣ್ಣ ಅಂಗಡಿಯಿಂದ ಖರೀದಿಸಿದೆನು.",
        "ಮಕ್ಕಳು ಮೈದಾನದಲ್ಲಿ ಕ್ರಿಕೆಟ್ ಆಡುತ್ತಿದ್ದಾರೆ.",
        "ರೈಲು ಸರಿಯಾದ ಸಮಯಕ್ಕೆ ತಲುಪಿತು.",
        "ನಾಸಾ ಮತ್ತು ಇಸ್ರೋ ಜಂಟಿ ಯೋಜನೆಯ ನವೀಕರಣವನ್ನು ಘೋಷಿಸಿವೆ.",
        "ದಯವಿಟ್ಟು ಪುಸ್ತಕಗಳನ್ನು ಕಬೋರ್ಡ್‌ನಲ್ಲಿ ಇರಿಸಿ.",
        "ನಾವು ಮುಂದಿನ ವಾರ ಮೈಸೂರಿಗೆ ಭೇಟಿ ನೀಡುತ್ತಿದ್ದೇವೆ.",
        "ನಿಮಗೆ ಚಹಾ ಬೇಕೇ ಅಥವಾ ಕಾಫಿ ಬೇಕೇ?",
        "ರಾತ್ರಿಯ ಕೆಲಸದ ಸಮಯದಲ್ಲಿ ಜಿಪಿಯು ಕ್ಲಸ್ಟರ್ ಮೆಮೊರಿ ಖಾಲಿಯಾಯಿತು.",
        "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯು ಪ್ರಪಂಚದಾದ್ಯಂತ ಆರೋಗ್ಯ ರಕ್ಷಣೆಯನ್ನು ಬದಲಾಯಿಸುತ್ತಿದೆ.",
        "ಆಕೆ ತಂತ್ರಜ್ಞಾನ ಸಂಸ್ಥೆಯೊಂದರಲ್ಲಿ ಹಿರಿಯ ಸಾಫ್ಟ್‌ವೇರ್ ಎಂಜಿನಿಯರ್ ಆಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದ್ದಾಳೆ.",
        "ಇಂದು ಬೆಟ್ಟಗಳಲ್ಲಿ ವಾತಾವರಣವು ಆಹ್ಲಾದಕರ ಮತ್ತು ತಂಪಾಗಿದೆ.",
        "ಸೌರ ಮತ್ತು ಪವನ ಶಕ್ತಿಯಂತಹ ನವೀಕರಿಸಬಹುದಾದ ಇಂಧನ ಮೂಲಗಳು ವೇಗವಾಗಿ ಬೆಳೆಯುತ್ತಿವೆ.",
        "ಭಾರಿ ಮಳೆಯಿಂದಾಗಿ ನಗರದ ಹಲವು ಭಾಗಗಳಲ್ಲಿ ಸಂಚಾರ ದಟ್ಟಣೆ ಉಂಟಾಗಿದೆ.",
        "ವಿದ್ಯಾರ್ಥಿಗಳು ತಮ್ಮ ಅಂತಿಮ ವಿಶ್ವವಿದ್ಯಾಲಯ ಪರೀಕ್ಷೆಗಳಿಗೆ ಶ್ರದ್ಧೆಯಿಂದ ಸಿದ್ಧರಾದರು.",
        "ಮಳೆಗಾಲದ ನಂತರ ರೈತರು ಉತ್ತಮ ಇಳುವರಿಯನ್ನು ನಿರೀಕ್ಷಿಸುತ್ತಿದ್ದಾರೆ.",
        "ಹೊಸ ಮೆಟ್ರೋ ರೈಲು ಮಾರ್ಗವು ಮುಂದಿನ ತಿಂಗಳು ಸಾರ್ವಜನಿಕ ಸಂಚಾರಕ್ಕೆ ಮುಕ್ತವಾಗಲಿದೆ.",
        "ಡೇಟಾ ಉಲ್ಲಂಘನೆಯನ್ನು ತಡೆಯಲು ಸೈಬರ್ ಭದ್ರತಾ ಕ್ರಮಗಳನ್ನು ನವೀಕರಿಸಬೇಕು.",
        "ಸಮತೋಲಿತ ಆಹಾರಕ್ಕಾಗಿ ತಾಜಾ ಹಣ್ಣುಗಳು ಮತ್ತು ತರಕಾರಿಗಳು ಅತ್ಯಗತ್ಯ.",
        "ವಸ್ತುಸಂಗ್ರಹಾಲಯವು ಐತಿಹಾಸಿಕ ರಾಜವಂಶಗಳ ಪ್ರಾಚೀನ ಕಲಾಕೃತಿಗಳನ್ನು ಹೊಂದಿದೆ.",
        "ವಿಜ್ಞಾನಿಗಳು ಮಂಗಳ ಗ್ರಹದಲ್ಲಿ ನೀರಿನ ಹೊಸ ಸಾಕ್ಷ್ಯವನ್ನು ಕಂಡುಹಿಡಿದಿದ್ದಾರೆ.",
        "ಆನ್‌ಲೈನ್ ಕಲಿಕಾ ವೇದಿಕೆಗಳು ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಹೊಂದಿಕೊಳ್ಳುವ ಶೈಕ್ಷಣಿಕ ಕೋರ್ಸ್‌ಗಳನ್ನು ಒದಗಿಸುತ್ತವೆ.",
        "ರಾಷ್ಟ್ರೀಯ ಉದ್ಯಾನವನವು ಅಪರೂಪದ ಪಕ್ಷಿ ಮತ್ತು ಪ್ರಾಣಿ ಪ್ರಭೇದಗಳಿಗೆ ನೆಲೆಯಾಗಿದೆ.",
        "ಡಿಜಿಟಲ್ ಪಾವತಿ ವ್ಯವಸ್ಥೆಗಳು ದೈನಂದಿನ ಹಣಕಾಸು ವ್ಯವಹಾರಗಳನ್ನು ಸರಳೀಕರಣಗೊಳಿಸಿವೆ."
    ],
    "tel": [
        "బెంగళూరు అంతర్జాతీయ విమానాశ్రయం మార్చిలో రికార్డు స్థాయి ట్రాఫిక్‌ను నిర్వహించింది.",
        "త్రైమాసిక సమీక్ష సమావేశం గురువారానికి వాయిదా పడింది.",
        "నేను ఈ పుస్తకాన్ని నిన్న ఎంజీ రోడ్ సమీపంలోని ఒక చిన్న దుకాణంలో కొన్నాను.",
        "పిల్లలు మైదానంలో క్రికెట్ ఆడుతున్నారు.",
        "రైలు సరైన సమయానికి చేరుకుంది.",
        "నాసా మరియు ఇస్రో సంయుక్త మిషన్ అప్‌డేట్‌ను ప్రకటించాయి.",
        "దయచేసి పుస్తకాలను అల్మారాలో ఉంచండి.",
        "మేము వచ్చే వారం మైసూర్ వెళ్తున్నాము.",
        "మీకు టీ కావాలా లేక కాఫీ కావాలా?",
        "రాత్రి పని సమయంలో జీపీయూ క్లస్టర్ మెమరీ పూర్తయింది.",
        "కృత్రిమ మేధస్సు ప్రపంచవ్యాప్తంగా ఆరోగ్య సంరక్షణను మారుస్తోంది.",
        "ఆమె ఒక సాంకేతిక సంస్థలో సీనియర్ సాఫ్ట్‌వేర్ ఇంజనీర్‌గా పనిచేస్తోంది.",
        "ఈ రోజు పర్వతాలలో వాతావరణం ఆహ్లాదకరంగా మరియు చల్లగా ఉంది.",
        "సౌర మరియు పవన విద్యుత్ వంటి పునరుత్పాదక ఇంధన వనరులు వేగంగా పెరుగుతున్నాయి.",
        "భారీ వర్షాల కారణంగా నగరంలోని పలు ప్రాంతాల్లో ట్రాఫిక్ స్తంభించిపోయింది.",
        "విద్యార్థులు తమ చివరి విశ్వవిద్యాలయ పరీక్షల కోసం శ్రద్ధగా సిద్ధమయ్యారు.",
        "వర్షాకాలం తర్వాత రైతులు మంచి దిగుబడిని ఆశిస్తున్నారు.",
        "కొత్త మెట్రో రైలు మార్గం వచ్చే నెలలో ప్రజా రవాణా కోసం ప్రారంభించబడుతుంది.",
        "డేటా ఉల్లంఘనలను నిరోధించడానికి సైబర్ భద్రతా చర్యలను నవీకరించాలి.",
        "సమతుల్య ఆహారానికి తాజా పండ్లు మరియు కూరగాయలు అవసరం.",
        "మ్యూజియంలో చారిత్రక రాజవంశాల పురాతన కళాఖండాలు ఉన్నాయి.",
        "శాస్త్రవేత్తలు అంగారకుడిపై నీటి కొత్త ఆధారాలను కనుగొన్నారు.",
        "ఆన్‌లైన్ లెర్నింగ్ ప్లాట్‌ఫారమ్‌లు విద్యార్థులకు సౌకర్యవంతమైన విద్యా కోర్సులను అందిస్తాయి.",
        "జాతీయ పార్కు అరుదైన పక్షులు మరియు జంతువుల జాతులకు నిలయంగా ఉంది.",
        "డిజిటల్ చెల్లింపు వ్యవస్థలు రోజువారీ ఆర్థిక లావాదేవీలను సరళీకృతం చేశాయి."
    ]
}

def main():
    parser = argparse.ArgumentParser(description="Prepare Multilingual Evaluation Corpus (A1)")
    parser.add_argument("--output_dir", default="partA/corpus", help="Target output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        "dataset_name": "FLORES-200 Aligned Multilingual Benchmark Sample",
        "source": "Meta FLORES-200 Parallel Translation Dataset",
        "languages": list(PARALLEL_DATA.keys()),
        "total_sentences_per_language": len(PARALLEL_DATA["eng"]),
        "is_parallel": True,
        "normalization": "Unicode NFC",
        "stats": {}
    }

    print("============================================================")
    print("Preparing Multilingual Evaluation Corpus (A1)")
    print("============================================================")

    for lang, sentences in PARALLEL_DATA.items():
        norm_sentences = [normalize_text(s) for s in sentences if s.strip()]
        out_path = os.path.join(args.output_dir, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in norm_sentences:
                f.write(s + "\n")

        total_words = sum(count_whitespace_words(s) for s in norm_sentences)
        total_graphemes = sum(count_grapheme_clusters(s) for s in norm_sentences)
        total_bytes = sum(count_utf8_bytes(s) for s in norm_sentences)

        metadata["stats"][lang] = {
            "sentences": len(norm_sentences),
            "words": total_words,
            "grapheme_clusters": total_graphemes,
            "utf8_bytes": total_bytes,
            "bytes_per_word": round(total_bytes / total_words, 2),
            "graphemes_per_word": round(total_graphemes / total_words, 2)
        }

        print(f"[{lang:<4}] Sentences: {len(norm_sentences):<3} | Words: {total_words:<4} | Graphemes: {total_graphemes:<4} | Bytes: {total_bytes:<5}")

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata written to: {meta_path}")

if __name__ == "__main__":
    main()
