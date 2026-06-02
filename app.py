import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from PIL import Image
from datetime import datetime
from treatment_info import TREATMENT_INFO
from gtts import gTTS
import tempfile
import requests
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

MODEL_PATH = "model/best_model.keras"
CLASS_PATH = "model/class_names.json"
HISTORY_PATH = "prediction_history.csv"
IMG_SIZE = (224, 224)

TEXT = {
    "English": {
        "title": "KrishiMitra AI",
        "subtitle": "AI-powered plant disease detection assistant for farmers",
        "upload": "Upload Leaf Image",
        "choose": "Choose a clear leaf image",
        "detect": "Detect Disease",
        "result": "Detection Result",
        "history": "History",
        "about": "About",
        "crop": "Crop",
        "condition": "Condition",
        "confidence": "Confidence",
        "top3": "Top 3 Predictions",
        "action": "Detailed Cure Guidance",
        "no_upload": "Upload or capture a leaf image to start detection.",
        "warning": "KrishiMitra AI is a prototype assistant and should not replace expert agricultural advice.",
        "speak": "Speak Result",
        "download": "Download AI Report",
        "accuracy": "Accuracy",
        "classes": "Classes",
        "images": "Images",
        "total_scans": "Total Scans",
        "camera": "Capture Leaf Image",
        "captured_image": "Captured Image",
        "severity": "Severity",
        "weather": "Weather Alert",
        "weather_title": "Weather-Based Disease Risk",
        "humidity": "Humidity",
        "temperature": "Temperature",
        "high_weather_risk": "High Risk: Humid weather can increase fungal diseases like blight, rust, and powdery mildew.",
        "high_weather_action": "Action: Avoid overhead watering, improve airflow, remove infected leaves, and inspect crops daily.",
        "medium_weather_risk": "Medium Risk: Conditions are suitable for some disease spread.",
        "medium_weather_action": "Action: Monitor leaves closely and avoid excess moisture around plants.",
        "low_weather_risk": "Low Risk: Current weather condition is less favorable for fungal disease spread.",
        "low_weather_action": "Action: Continue regular monitoring and good plant care.",
        "weather_note": "Note: This is a basic weather-risk estimator. Future version can connect to a live weather API.",
        "ask_ai": "Ask KrishiMitra",
        "ask_desc": "Ask basic farming questions about watering, disease spread, fungicide, prevention, and crop care.",
        "ask_placeholder": "Type your farming question here",
        "ans_spread": "Many plant diseases spread faster in humid weather, through water splash, infected leaves, tools, and close plant spacing. Remove infected leaves and avoid overhead watering.",
        "ans_water": "Avoid overwatering. Water near the roots instead of wetting leaves, because wet leaves increase fungal disease risk.",
        "ans_fertilizer": "Use balanced fertilizer based on crop stage. Excess nitrogen can increase soft growth and disease risk.",
        "ans_fungicide": "Use fungicide only after identifying the disease correctly. Prefer expert advice before chemical use. Organic methods like pruning infected leaves and neem spray may help in early stages.",
        "ans_prevention": "Good prevention includes crop rotation, proper spacing, clean tools, removing infected leaves, avoiding water splash, and regular leaf inspection.",
        "ans_expert": "Contact a local agriculture expert or Krishi Vigyan Kendra if infection spreads quickly, fruits are affected, or more than 20–30% leaves show symptoms.",
        "ans_default": "I can answer basic questions about disease spread, watering, fertilizer, fungicide, prevention, and expert help. For serious crop loss, consult a local agriculture expert.",
        "unique_diseases": "Unique Diseases",
        "most_common": "Most Common",
        "disease_distribution": "Disease Distribution",
        "prediction_history": "Prediction History",
        "no_history": "No prediction history yet.",
        "status": "Status",
        "cause": "Cause",
        "symptoms": "Symptoms",
        "spread": "How it spreads",
        "organic": "Organic method",
        "chemical": "Chemical method",
        "prevention": "Prevention",
        "expert": "When to contact expert",
        "risk_level": "Risk Level",
        "low_priority": "Plant condition appears stable.",
        "medium_priority": "Start treatment and monitor daily.",
        "high_priority": "Immediate treatment required within 24–48 hours.",
        "about_title": "About KrishiMitra AI",
        "about_para": "KrishiMitra AI is an AI-powered plant disease detection assistant designed to help farmers identify crop diseases using a simple leaf image. The system uses EfficientNetB0 trained on the PlantVillage dataset with 38 disease classes and 54,305+ images.",
        "features": "App Features",
        "features_list": "- Disease Detection\n- Camera Capture and Image Upload\n- Top 3 Predictions\n- Detailed Cure Guidance\n- Organic and Chemical Treatment Suggestions\n- Action Priority Alerts\n- History Tracking and Disease Analytics\n- Weather-Based Disease Risk\n- Ask KrishiMitra Assistant\n- Multi-language UI\n- Farmer-friendly Interface\n- Voice Result Output\n- PDF Report Download",
        "technology": "Technology",
        "technology_list": "- Python\n- TensorFlow / Keras\n- EfficientNetB0 Transfer Learning\n- Streamlit Web App\n- PlantVillage Dataset",
        "performance": "Current Performance",
        "performance_list": "- Dataset Images: 54,305+\n- Classes: 38\n- Validation Accuracy: 94.74%",
        "made_by": "Made by Krishna Saini"
    },
    "Hindi": {
        "title": "कृषिमित्र AI",
        "subtitle": "किसानों के लिए AI आधारित पौध रोग पहचान सहायक",
        "upload": "पत्ती की तस्वीर अपलोड करें",
        "choose": "साफ पत्ती की तस्वीर चुनें",
        "detect": "रोग पहचानें",
        "result": "पहचान परिणाम",
        "history": "इतिहास",
        "about": "परिचय",
        "crop": "फसल",
        "condition": "स्थिति",
        "confidence": "विश्वास स्तर",
        "top3": "शीर्ष 3 अनुमान",
        "action": "विस्तृत उपचार जानकारी",
        "no_upload": "पहचान शुरू करने के लिए पत्ती की तस्वीर अपलोड या कैप्चर करें।",
        "warning": "कृषिमित्र AI एक प्रोटोटाइप सहायक है, यह कृषि विशेषज्ञ की सलाह का विकल्प नहीं है।",
        "speak": "परिणाम सुनें",
        "download": "AI रिपोर्ट डाउनलोड करें",
        "accuracy": "सटीकता",
        "classes": "क्लास",
        "images": "तस्वीरें",
        "total_scans": "कुल स्कैन",
        "camera": "पत्ती की फोटो लें",
        "captured_image": "खींची गई तस्वीर",
        "severity": "गंभीरता",
        "weather": "मौसम चेतावनी",
        "weather_title": "मौसम आधारित रोग जोखिम",
        "humidity": "आर्द्रता",
        "temperature": "तापमान",
        "high_weather_risk": "उच्च जोखिम: नमी वाला मौसम ब्लाइट, रस्ट और पाउडरी मिल्ड्यू जैसे फंगल रोग बढ़ा सकता है।",
        "high_weather_action": "कार्य: ऊपर से पानी देने से बचें, हवा का प्रवाह सुधारें, संक्रमित पत्तियाँ हटाएँ और रोज़ निरीक्षण करें।",
        "medium_weather_risk": "मध्यम जोखिम: कुछ रोग फैलने के लिए मौसम अनुकूल हो सकता है।",
        "medium_weather_action": "कार्य: पत्तियों को ध्यान से देखें और पौधों के आसपास अधिक नमी से बचें।",
        "low_weather_risk": "कम जोखिम: वर्तमान मौसम फंगल रोग फैलने के लिए कम अनुकूल है।",
        "low_weather_action": "कार्य: नियमित निगरानी और अच्छी देखभाल जारी रखें।",
        "weather_note": "नोट: यह एक बेसिक मौसम-जोखिम अनुमानक है। भविष्य में इसे लाइव मौसम API से जोड़ा जा सकता है।",
        "ask_ai": "कृषिमित्र से पूछें",
        "ask_desc": "पानी, रोग फैलाव, फंगीसाइड, बचाव और फसल देखभाल से जुड़े सवाल पूछें।",
        "ask_placeholder": "अपना कृषि प्रश्न यहाँ लिखें",
        "ans_spread": "कई पौध रोग नमी में, पानी के छींटों, संक्रमित पत्तियों, औजारों और कम दूरी वाले पौधों से तेजी से फैलते हैं। संक्रमित पत्तियाँ हटाएँ और ऊपर से पानी देने से बचें।",
        "ans_water": "अधिक पानी देने से बचें। पत्तियों को गीला करने की जगह जड़ों के पास पानी दें।",
        "ans_fertilizer": "फसल की अवस्था के अनुसार संतुलित खाद दें। अधिक नाइट्रोजन से रोग का जोखिम बढ़ सकता है।",
        "ans_fungicide": "रोग की सही पहचान के बाद ही फंगीसाइड का उपयोग करें। रसायन से पहले विशेषज्ञ की सलाह लें।",
        "ans_prevention": "अच्छे बचाव में फसल चक्र, उचित दूरी, साफ औजार, संक्रमित पत्तियाँ हटाना और नियमित निरीक्षण शामिल हैं।",
        "ans_expert": "यदि रोग तेजी से फैल रहा है, फल प्रभावित हैं या 20–30% से अधिक पत्तियाँ संक्रमित हैं तो कृषि विशेषज्ञ या KVK से संपर्क करें।",
        "ans_default": "मैं रोग फैलाव, पानी, खाद, फंगीसाइड, बचाव और विशेषज्ञ सहायता पर बेसिक जवाब दे सकता हूँ। गंभीर नुकसान में विशेषज्ञ से सलाह लें।",
        "unique_diseases": "अलग-अलग रोग",
        "most_common": "सबसे सामान्य",
        "disease_distribution": "रोग वितरण",
        "prediction_history": "पहचान इतिहास",
        "no_history": "अभी कोई पहचान इतिहास नहीं है।",
        "status": "स्थिति",
        "cause": "कारण",
        "symptoms": "लक्षण",
        "spread": "कैसे फैलता है",
        "organic": "जैविक तरीका",
        "chemical": "रासायनिक तरीका",
        "prevention": "बचाव",
        "expert": "विशेषज्ञ से कब संपर्क करें",
        "risk_level": "जोखिम स्तर",
        "low_priority": "पौधे की स्थिति स्थिर लग रही है।",
        "medium_priority": "उपचार शुरू करें और रोज़ निगरानी करें।",
        "high_priority": "24–48 घंटों के अंदर तुरंत उपचार की आवश्यकता है।",
        "about_title": "कृषिमित्र AI के बारे में",
        "about_para": "कृषिमित्र AI एक AI आधारित पौध रोग पहचान सहायक है जो पत्ती की तस्वीर से किसानों को फसल रोग पहचानने में मदद करता है। यह सिस्टम PlantVillage dataset की 54,305+ तस्वीरों और 38 रोग क्लास पर प्रशिक्षित EfficientNetB0 मॉडल का उपयोग करता है।",
        "features": "ऐप फीचर्स",
        "features_list": "- रोग पहचान\n- कैमरा कैप्चर और इमेज अपलोड\n- शीर्ष 3 अनुमान\n- विस्तृत उपचार मार्गदर्शन\n- जैविक और रासायनिक उपचार सुझाव\n- कार्य प्राथमिकता अलर्ट\n- इतिहास और रोग विश्लेषण\n- मौसम आधारित रोग जोखिम\n- कृषिमित्र सहायक\n- बहुभाषी UI\n- किसान अनुकूल इंटरफेस\n- वॉइस आउटपुट\n- PDF रिपोर्ट डाउनलोड",
        "technology": "तकनीक",
        "technology_list": "- Python\n- TensorFlow / Keras\n- EfficientNetB0 Transfer Learning\n- Streamlit Web App\n- PlantVillage Dataset",
        "performance": "वर्तमान प्रदर्शन",
        "performance_list": "- Dataset Images: 54,305+\n- Classes: 38\n- Validation Accuracy: 94.74%",
        "made_by": "कृष्णा सैनी द्वारा निर्मित"
    },
    "Punjabi": {
        "title": "ਕ੍ਰਿਸ਼ੀਮਿਤਰ AI",
        "subtitle": "ਕਿਸਾਨਾਂ ਲਈ AI ਆਧਾਰਿਤ ਪੌਧਾ ਰੋਗ ਪਛਾਣ ਸਹਾਇਕ",
        "upload": "ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
        "choose": "ਸਾਫ਼ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਚੁਣੋ",
        "detect": "ਰੋਗ ਪਛਾਣੋ",
        "result": "ਪਛਾਣ ਨਤੀਜਾ",
        "history": "ਇਤਿਹਾਸ",
        "about": "ਜਾਣਕਾਰੀ",
        "crop": "ਫਸਲ",
        "condition": "ਹਾਲਤ",
        "confidence": "ਭਰੋਸਾ",
        "top3": "ਸਭ ਤੋਂ ਉੱਪਰਲੇ 3 ਅਨੁਮਾਨ",
        "action": "ਵਿਸਥਾਰ ਇਲਾਜ ਜਾਣਕਾਰੀ",
        "no_upload": "ਪਛਾਣ ਸ਼ੁਰੂ ਕਰਨ ਲਈ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਜਾਂ ਕੈਪਚਰ ਕਰੋ।",
        "warning": "ਕ੍ਰਿਸ਼ੀਮਿਤਰ AI ਇੱਕ ਪ੍ਰੋਟੋਟਾਈਪ ਸਹਾਇਕ ਹੈ, ਇਹ ਖੇਤੀ ਮਾਹਿਰ ਦੀ ਸਲਾਹ ਦਾ ਬਦਲ ਨਹੀਂ ਹੈ।",
        "speak": "ਨਤੀਜਾ ਸੁਣੋ",
        "download": "AI ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ",
        "accuracy": "ਸਹੀਤਾ",
        "classes": "ਕਲਾਸਾਂ",
        "images": "ਤਸਵੀਰਾਂ",
        "total_scans": "ਕੁੱਲ ਸਕੈਨ",
        "camera": "ਪੱਤੇ ਦੀ ਫੋਟੋ ਖਿੱਚੋ",
        "captured_image": "ਖਿੱਚੀ ਤਸਵੀਰ",
        "severity": "ਗੰਭੀਰਤਾ",
        "weather": "ਮੌਸਮ ਚੇਤਾਵਨੀ",
        "weather_title": "ਮੌਸਮ ਆਧਾਰਿਤ ਰੋਗ ਖਤਰਾ",
        "humidity": "ਨਮੀ",
        "temperature": "ਤਾਪਮਾਨ",
        "high_weather_risk": "ਉੱਚ ਖਤਰਾ: ਨਮੀ ਵਾਲਾ ਮੌਸਮ ਫੰਗਲ ਰੋਗ ਵਧਾ ਸਕਦਾ ਹੈ।",
        "high_weather_action": "ਕਾਰਵਾਈ: ਉੱਪਰੋਂ ਪਾਣੀ ਨਾ ਦਿਓ, ਹਵਾ ਦਾ ਪ੍ਰਵਾਹ ਸੁਧਾਰੋ ਅਤੇ ਸੰਕਰਮਿਤ ਪੱਤੇ ਹਟਾਓ।",
        "medium_weather_risk": "ਦਰਮਿਆਨਾ ਖਤਰਾ: ਕੁਝ ਰੋਗ ਫੈਲਣ ਲਈ ਹਾਲਾਤ ਅਨੁਕੂਲ ਹਨ।",
        "medium_weather_action": "ਕਾਰਵਾਈ: ਪੱਤਿਆਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ ਅਤੇ ਵਾਧੂ ਨਮੀ ਤੋਂ ਬਚੋ।",
        "low_weather_risk": "ਘੱਟ ਖਤਰਾ: ਮੌਸਮ ਫੰਗਲ ਰੋਗ ਲਈ ਘੱਟ ਅਨੁਕੂਲ ਹੈ।",
        "low_weather_action": "ਕਾਰਵਾਈ: ਨਿਯਮਤ ਦੇਖਭਾਲ ਜਾਰੀ ਰੱਖੋ।",
        "weather_note": "ਨੋਟ: ਇਹ ਬੇਸਿਕ ਮੌਸਮ-ਖਤਰਾ ਅਨੁਮਾਨ ਹੈ। ਭਵਿੱਖ ਵਿੱਚ live weather API ਜੋੜ ਸਕਦੇ ਹਾਂ।",
        "ask_ai": "ਕ੍ਰਿਸ਼ੀਮਿਤਰ ਨੂੰ ਪੁੱਛੋ",
        "ask_desc": "ਪਾਣੀ, ਰੋਗ ਫੈਲਾਅ, ਫੰਗੀਸਾਈਡ, ਬਚਾਅ ਅਤੇ ਫਸਲ ਦੇਖਭਾਲ ਬਾਰੇ ਪੁੱਛੋ।",
        "ask_placeholder": "ਆਪਣਾ ਖੇਤੀ ਪ੍ਰਸ਼ਨ ਲਿਖੋ",
        "ans_spread": "ਕਈ ਪੌਧੇ ਦੇ ਰੋਗ ਨਮੀ, ਪਾਣੀ ਦੇ ਛਿਡਕਾਅ, ਸੰਕਰਮਿਤ ਪੱਤਿਆਂ ਅਤੇ ਔਜ਼ਾਰਾਂ ਰਾਹੀਂ ਤੇਜ਼ੀ ਨਾਲ ਫੈਲਦੇ ਹਨ।",
        "ans_water": "ਜ਼ਿਆਦਾ ਪਾਣੀ ਦੇਣ ਤੋਂ ਬਚੋ। ਪੱਤਿਆਂ ਦੀ ਥਾਂ ਜੜਾਂ ਦੇ ਕੋਲ ਪਾਣੀ ਦਿਓ।",
        "ans_fertilizer": "ਫਸਲ ਦੀ ਲੋੜ ਅਨੁਸਾਰ ਸੰਤੁਲਿਤ ਖਾਦ ਵਰਤੋ। ਵੱਧ ਨਾਈਟ੍ਰੋਜਨ ਰੋਗ ਦਾ ਖਤਰਾ ਵਧਾ ਸਕਦੀ ਹੈ।",
        "ans_fungicide": "ਰੋਗ ਦੀ ਸਹੀ ਪਛਾਣ ਤੋਂ ਬਾਅਦ ਹੀ ਫੰਗੀਸਾਈਡ ਵਰਤੋ। ਰਸਾਇਣ ਤੋਂ ਪਹਿਲਾਂ ਮਾਹਿਰ ਦੀ ਸਲਾਹ ਲਵੋ।",
        "ans_prevention": "ਫਸਲ ਚੱਕਰ, ਠੀਕ ਦੂਰੀ, ਸਾਫ਼ ਔਜ਼ਾਰ ਅਤੇ ਨਿਯਮਤ ਜਾਂਚ ਚੰਗੇ ਬਚਾਅ ਹਨ।",
        "ans_expert": "ਜੇ ਰੋਗ ਤੇਜ਼ੀ ਨਾਲ ਫੈਲਦਾ ਹੈ ਜਾਂ 20–30% ਪੱਤੇ ਪ੍ਰਭਾਵਿਤ ਹਨ ਤਾਂ ਖੇਤੀ ਮਾਹਿਰ ਜਾਂ KVK ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        "ans_default": "ਮੈਂ ਰੋਗ ਫੈਲਾਅ, ਪਾਣੀ, ਖਾਦ, ਫੰਗੀਸਾਈਡ ਅਤੇ ਬਚਾਅ ਬਾਰੇ ਬੇਸਿਕ ਜਵਾਬ ਦੇ ਸਕਦਾ ਹਾਂ।",
        "unique_diseases": "ਵੱਖ-ਵੱਖ ਰੋਗ",
        "most_common": "ਸਭ ਤੋਂ ਆਮ",
        "disease_distribution": "ਰੋਗ ਵੰਡ",
        "prediction_history": "ਪਛਾਣ ਇਤਿਹਾਸ",
        "no_history": "ਅਜੇ ਕੋਈ ਪਛਾਣ ਇਤਿਹਾਸ ਨਹੀਂ।",
        "status": "ਸਥਿਤੀ",
        "cause": "ਕਾਰਣ",
        "symptoms": "ਲੱਛਣ",
        "spread": "ਕਿਵੇਂ ਫੈਲਦਾ ਹੈ",
        "organic": "ਜੈਵਿਕ ਤਰੀਕਾ",
        "chemical": "ਰਸਾਇਣਕ ਤਰੀਕਾ",
        "prevention": "ਬਚਾਅ",
        "expert": "ਮਾਹਿਰ ਨਾਲ ਕਦੋਂ ਸੰਪਰਕ ਕਰੋ",
        "risk_level": "ਖਤਰਾ ਪੱਧਰ",
        "low_priority": "ਪੌਧੇ ਦੀ ਹਾਲਤ ਸਥਿਰ ਲੱਗਦੀ ਹੈ।",
        "medium_priority": "ਇਲਾਜ ਸ਼ੁਰੂ ਕਰੋ ਅਤੇ ਰੋਜ਼ਾਨਾ ਨਿਗਰਾਨੀ ਕਰੋ।",
        "high_priority": "24–48 ਘੰਟਿਆਂ ਵਿੱਚ ਤੁਰੰਤ ਇਲਾਜ ਦੀ ਲੋੜ ਹੈ।",
        "about_title": "ਕ੍ਰਿਸ਼ੀਮਿਤਰ AI ਬਾਰੇ",
        "about_para": "ਕ੍ਰਿਸ਼ੀਮਿਤਰ AI ਇੱਕ AI ਆਧਾਰਿਤ ਪੌਧਾ ਰੋਗ ਪਛਾਣ ਸਹਾਇਕ ਹੈ ਜੋ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਨਾਲ ਕਿਸਾਨਾਂ ਦੀ ਮਦਦ ਕਰਦਾ ਹੈ।",
        "features": "ਐਪ ਫੀਚਰ",
        "features_list": "- ਰੋਗ ਪਛਾਣ\n- ਕੈਮਰਾ ਕੈਪਚਰ ਅਤੇ ਤਸਵੀਰ ਅੱਪਲੋਡ\n- ਟਾਪ 3 ਅਨੁਮਾਨ\n- ਵਿਸਥਾਰ ਇਲਾਜ ਜਾਣਕਾਰੀ\n- ਜੈਵਿਕ ਅਤੇ ਰਸਾਇਣਕ ਇਲਾਜ ਸੁਝਾਅ\n- ਕਾਰਵਾਈ ਪ੍ਰਾਥਮਿਕਤਾ ਚੇਤਾਵਨੀ\n- ਇਤਿਹਾਸ ਅਤੇ ਰੋਗ ਵਿਸ਼ਲੇਸ਼ਣ\n- ਮੌਸਮ ਆਧਾਰਿਤ ਰੋਗ ਖਤਰਾ\n- ਕ੍ਰਿਸ਼ੀਮਿਤਰ ਸਹਾਇਕ\n- ਬਹੁਭਾਸ਼ੀ UI\n- ਕਿਸਾਨ-ਮਿੱਤਰ ਇੰਟਰਫੇਸ\n- ਵੌਇਸ ਨਤੀਜਾ\n- PDF ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ",
        "technology": "ਤਕਨੀਕ",
        "technology_list": "- Python\n- TensorFlow / Keras\n- EfficientNetB0 Transfer Learning\n- Streamlit Web App\n- PlantVillage Dataset",
        "performance": "ਮੌਜੂਦਾ ਪ੍ਰਦਰਸ਼ਨ",
        "performance_list": "- Dataset Images: 54,305+\n- Classes: 38\n- Validation Accuracy: 94.74%",
        "made_by": "ਕ੍ਰਿਸ਼ਨਾ ਸੈਨੀ ਵੱਲੋਂ ਬਣਾਇਆ ਗਿਆ"
    },
    "Tamil": {
        "title": "கிருஷிமித்ரா AI",
        "subtitle": "விவசாயிகளுக்கான AI அடிப்படையிலான தாவர நோய் கண்டறிதல் உதவியாளர்",
        "upload": "இலை படத்தை பதிவேற்றவும்",
        "choose": "தெளிவான இலை படத்தைத் தேர்வு செய்யவும்",
        "detect": "நோயைக் கண்டறி",
        "result": "கண்டறிதல் முடிவு",
        "history": "வரலாறு",
        "about": "பற்றி",
        "crop": "பயிர்",
        "condition": "நிலை",
        "confidence": "நம்பிக்கை",
        "top3": "முதல் 3 கணிப்புகள்",
        "action": "விரிவான சிகிச்சை வழிகாட்டல்",
        "no_upload": "கண்டறிதலை தொடங்க இலை படத்தை பதிவேற்றவும் அல்லது படம் எடுக்கவும்.",
        "warning": "கிருஷிமித்ரா AI ஒரு மாதிரி உதவியாளர் மட்டுமே; இது நிபுணர் ஆலோசனையை மாற்றாது.",
        "speak": "முடிவை கேளுங்கள்",
        "download": "AI அறிக்கையை பதிவிறக்கவும்",
        "accuracy": "துல்லியம்",
        "classes": "வகைகள்",
        "images": "படங்கள்",
        "total_scans": "மொத்த ஸ்கேன்",
        "camera": "இலை படத்தை எடுக்கவும்",
        "captured_image": "எடுக்கப்பட்ட படம்",
        "severity": "தீவிரம்",
        "weather": "வானிலை எச்சரிக்கை",
        "weather_title": "வானிலை அடிப்படையிலான நோய் அபாயம்",
        "humidity": "ஈரப்பதம்",
        "temperature": "வெப்பநிலை",
        "high_weather_risk": "அதிக அபாயம்: ஈரப்பதமான வானிலை பிளைட், ரஸ்ட், பவுடரி மில்டியூ போன்ற பூஞ்சை நோய்களை அதிகரிக்கலாம்.",
        "high_weather_action": "செயல்: மேலிருந்து நீர் ஊற்றாதீர்கள், காற்றோட்டத்தை மேம்படுத்துங்கள், பாதிக்கப்பட்ட இலைகளை அகற்றுங்கள்.",
        "medium_weather_risk": "மிதமான அபாயம்: சில நோய்கள் பரவ ஏற்ற சூழல் உள்ளது.",
        "medium_weather_action": "செயல்: இலைகளை நெருக்கமாக கண்காணிக்கவும், அதிக ஈரப்பதத்தை தவிர்க்கவும்.",
        "low_weather_risk": "குறைந்த அபாயம்: தற்போதைய வானிலை பூஞ்சை நோய்க்கு குறைவான வாய்ப்பு அளிக்கிறது.",
        "low_weather_action": "செயல்: வழக்கமான பராமரிப்பு தொடரவும்.",
        "weather_note": "குறிப்பு: இது ஒரு அடிப்படை வானிலை-அபாய கணிப்பான். அடுத்த பதிப்பில் live weather API சேர்க்கலாம்.",
        "ask_ai": "கிருஷிமித்ராவிடம் கேளுங்கள்",
        "ask_desc": "நீர், நோய் பரவல், பூஞ்சை மருந்து, தடுப்பு மற்றும் பயிர் பராமரிப்பு பற்றி கேளுங்கள்.",
        "ask_placeholder": "உங்கள் விவசாய கேள்வியை இங்கே எழுதுங்கள்",
        "ans_spread": "பல தாவர நோய்கள் ஈரப்பதமான வானிலையில், நீர் தெறிப்பு, பாதிக்கப்பட்ட இலைகள் மற்றும் கருவிகள் மூலம் வேகமாக பரவுகின்றன.",
        "ans_water": "அதிக நீர் ஊற்றாதீர்கள். இலைகளை ஈரமாக்காமல் வேர் அருகே நீர் ஊற்றவும்.",
        "ans_fertilizer": "பயிர் நிலைக்கு ஏற்ப சமநிலை உரம் பயன்படுத்தவும். அதிக நைட்ரஜன் நோய் அபாயத்தை அதிகரிக்கலாம்.",
        "ans_fungicide": "நோயை சரியாக கண்டறிந்த பிறகு மட்டுமே பூஞ்சை மருந்து பயன்படுத்தவும். ரசாயனத்திற்கு முன் நிபுணரிடம் ஆலோசனை பெறவும்.",
        "ans_prevention": "பயிர் சுழற்சி, சரியான இடைவெளி, சுத்தமான கருவிகள், பாதிக்கப்பட்ட இலைகளை அகற்றுதல் மற்றும் வழக்கமான கண்காணிப்பு நல்ல தடுப்பு முறைகள்.",
        "ans_expert": "நோய் வேகமாக பரவினால் அல்லது 20–30% இலைகள் பாதிக்கப்பட்டால் விவசாய நிபுணர் அல்லது KVK-ஐ தொடர்பு கொள்ளுங்கள்.",
        "ans_default": "நோய் பரவல், நீர், உரம், பூஞ்சை மருந்து, தடுப்பு மற்றும் நிபுணர் உதவி பற்றி நான் அடிப்படை பதில் அளிக்க முடியும்.",
        "unique_diseases": "தனித்த நோய்கள்",
        "most_common": "அதிகம் காணப்பட்டது",
        "disease_distribution": "நோய் பகிர்வு",
        "prediction_history": "கண்டறிதல் வரலாறு",
        "no_history": "இன்னும் கண்டறிதல் வரலாறு இல்லை.",
        "status": "நிலை",
        "cause": "காரணம்",
        "symptoms": "அறிகுறிகள்",
        "spread": "எப்படி பரவுகிறது",
        "organic": "இயற்கை முறை",
        "chemical": "ரசாயன முறை",
        "prevention": "தடுப்பு",
        "expert": "நிபுணரை எப்போது தொடர்பு கொள்ள வேண்டும்",
        "risk_level": "அபாய நிலை",
        "low_priority": "தாவர நிலை நிலையானதாக தெரிகிறது.",
        "medium_priority": "சிகிச்சையைத் தொடங்கி தினமும் கண்காணிக்கவும்.",
        "high_priority": "24–48 மணி நேரத்திற்குள் உடனடி சிகிச்சை தேவை.",
        "about_title": "கிருஷிமித்ரா AI பற்றி",
        "about_para": "கிருஷிமித்ரா AI என்பது இலை படத்தை பயன்படுத்தி விவசாயிகளுக்கு பயிர் நோய்களை கண்டறிய உதவும் AI அடிப்படையிலான உதவியாளர் ஆகும்.",
        "features": "ஆப் அம்சங்கள்",
        "features_list": "- நோய் கண்டறிதல்\n- கேமரா படம் மற்றும் படப் பதிவேற்றம்\n- முதல் 3 கணிப்புகள்\n- விரிவான சிகிச்சை வழிகாட்டல்\n- இயற்கை மற்றும் ரசாயன சிகிச்சை பரிந்துரைகள்\n- செயல் முன்னுரிமை எச்சரிக்கை\n- வரலாறு மற்றும் நோய் பகுப்பாய்வு\n- வானிலை அடிப்படையிலான நோய் அபாயம்\n- கிருஷிமித்ரா உதவியாளர்\n- பல மொழி UI\n- விவசாயி நட்பு இடைமுகம்\n- குரல் முடிவு\n- PDF அறிக்கை பதிவிறக்கம்",
        "technology": "தொழில்நுட்பம்",
        "technology_list": "- Python\n- TensorFlow / Keras\n- EfficientNetB0 Transfer Learning\n- Streamlit Web App\n- PlantVillage Dataset",
        "performance": "தற்போதைய செயல்திறன்",
        "performance_list": "- Dataset Images: 54,305+\n- Classes: 38\n- Validation Accuracy: 94.74%",
        "made_by": "கிருஷ்ணா சைனி உருவாக்கியது"
    }
}

st.set_page_config(page_title="KrishiMitra AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(3, 20, 10, 0.88), rgba(3, 20, 10, 0.94)),
        url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #E8F5E9;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #021b0f, #063b20);
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem !important;
}

.main-title {
    font-size: 54px;
    font-weight: 900;
    color: #9be15d;
    animation: glow 2s ease-in-out infinite alternate;
}

.sub-title {
    font-size: 21px;
    color: #d7ffd9;
    margin-bottom: 20px;
}

@keyframes glow {
    from { text-shadow: 0 0 8px #4caf50; }
    to { text-shadow: 0 0 22px #a5d6a7; }
}

.glass-card {
    padding: 25px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(200, 255, 200, 0.18);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}

.result-card {
    padding: 25px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(76,175,80,0.25), rgba(0,80,40,0.30));
    border-left: 8px solid #9be15d;
}

.treatment-card {
    padding: 25px;
    border-radius: 22px;
    background: rgba(255, 243, 205, 0.15);
    border-left: 8px solid #ffd54f;
}

.about-card {
    padding: 32px;
    border-radius: 24px;
    background: rgba(0, 35, 18, 0.72);
    border: 1px solid rgba(165, 214, 167, 0.35);
    animation: floatIn 0.8s ease-out;
}

@keyframes floatIn {
    from { opacity: 0; transform: translateY(25px); }
    to { opacity: 1; transform: translateY(0); }
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #43a047, #9be15d);
    color: #062b13;
    font-weight: 800;
    border-radius: 14px;
    border: none;
    padding: 0.75rem 1rem;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #9be15d, #43a047);
    color: black;
}

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: rgba(0, 30, 15, 0.85);
    border: 1px solid rgba(155, 225, 93, 0.45);
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stFileUploader"] button, [data-testid="stCameraInput"] button {
    background: #064d25 !important;
    color: #e8f5e9 !important;
    border-radius: 12px !important;
    border: 1px solid #9be15d !important;
}

h1, h2, h3, h4, p, label, span {
    color: #E8F5E9 !important;
}

[data-testid="stMetricValue"] {
    color: #9be15d !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def split_crop_disease(name):
    parts = name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return crop, disease


def clean_name(name):
    return name.replace("___", " - ").replace("_", " ")


def get_treatment(predicted_class):
    for key, value in TREATMENT_INFO.items():
        if key.lower() in predicted_class.lower():
            return value

    return {
        "status": "Detected",
        "cause": "The exact cause is not available in the current treatment database.",
        "symptoms": "Symptoms may vary depending on crop and disease stage.",
        "spread": "It may spread depending on disease type, weather, and field condition.",
        "organic": "Remove affected leaves, keep the plant clean, and avoid water splash.",
        "chemical": "Consult a local agriculture expert before using any chemical treatment.",
        "prevention": "Maintain proper watering, spacing, field hygiene, and regular monitoring.",
        "expert": "Contact an agriculture expert for accurate treatment guidance.",
        "risk": "Medium"
    }


def normalize_treatment(treatment):
    return {
        "status": treatment.get("status", "Detected"),
        "cause": treatment.get("cause", "Cause information is not available yet."),
        "symptoms": treatment.get("symptoms", "Symptoms information is not available yet."),
        "spread": treatment.get("spread", "Spread information is not available yet."),
        "organic": treatment.get("organic", "Remove affected leaves and keep the plant clean."),
        "chemical": treatment.get("chemical", "Consult an agriculture expert before using chemicals."),
        "prevention": treatment.get("prevention", "Maintain regular monitoring and field hygiene."),
        "expert": treatment.get("expert", "Contact an agriculture expert if the condition worsens."),
        "risk": treatment.get("risk", "Medium")
    }


def save_history(crop, disease, confidence):
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Crop": crop,
        "Disease": disease,
        "Confidence": f"{confidence * 100:.2f}%"
    }

    try:
        df = pd.read_csv(HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)


def get_total_scans():
    try:
        return len(pd.read_csv(HISTORY_PATH))
    except Exception:
        return 0


def get_severity(confidence):
    if confidence > 0.95:
        return "🟢 Low Risk"
    if confidence > 0.85:
        return "🟡 Medium Risk"
    return "🔴 High Risk"


def show_action_priority(severity, T):
    if "High" in severity:
        st.error(f"🚨 {T['high_priority']}")
    elif "Medium" in severity:
        st.warning(f"⚠ {T['medium_priority']}")
    else:
        st.success(f"✅ {T['low_priority']}")


def speak_text(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Voice generation failed: {e}")


def create_pdf_report(crop, disease, confidence, severity, treatment):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle("KrishiMitra AI Report")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 60, "KrishiMitra AI - Plant Disease Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 100, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.drawString(50, height - 125, "Made by: Krishna Saini")

    y = height - 170

    report_data = [
        ("Crop", crop),
        ("Condition / Disease", disease),
        ("Confidence", f"{confidence * 100:.2f}%"),
        ("Severity", severity),
        ("Status", treatment["status"]),
        ("Cause", treatment["cause"]),
        ("Symptoms", treatment["symptoms"]),
        ("How it spreads", treatment["spread"]),
        ("Organic method", treatment["organic"]),
        ("Chemical method", treatment["chemical"]),
        ("Prevention", treatment["prevention"]),
        ("When to contact expert", treatment["expert"]),
        ("Risk Level", treatment["risk"]),
    ]

    for title, value in report_data:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, f"{title}:")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(200, y, str(value)[:90])
        y -= 25

        if y < 80:
            pdf.showPage()
            y = height - 60

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(50, 40, "Disclaimer: This is an AI prototype and should not replace expert agricultural advice.")

    pdf.save()
    buffer.seek(0)
    return buffer



def get_user_city():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        return data.get("city")
    except Exception:
        return None


def get_live_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return None, "OpenWeather API key not found."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return None, data.get("message", "Unable to fetch weather data.")

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"]
        }, None

    except Exception as e:
        return None, str(e)


def predict_weather_disease_risk(temperature, humidity):
    if humidity >= 80 and 18 <= temperature <= 32:
        return "🔴 High Risk", "High humidity may increase fungal diseases like blight, rust, and powdery mildew.", "Avoid overhead watering, improve airflow, and inspect crops daily."

    elif humidity >= 60 and 18 <= temperature <= 35:
        return "🟡 Medium Risk", "Weather may support some disease spread.", "Monitor leaves regularly and avoid excess moisture."

    else:
        return "🟢 Low Risk", "Current weather is less favorable for fungal disease spread.", "Continue normal care and weekly monitoring."


model = load_ai_model()
class_names = load_class_names()

with st.sidebar:
    st.title("🌱 KrishiMitra AI")

    language = st.selectbox(
        "🌐 Select Language",
        ["English", "Hindi", "Punjabi", "Tamil"]
    )

    T = TEXT[language]

    st.divider()
    st.metric("Model Accuracy", "94.74%")
    st.metric("Classes", len(class_names))
    st.metric("Dataset Images", "54,305+")

st.markdown(f'<div class="main-title">🌱 {T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{T["subtitle"]}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(f"🎯 {T['accuracy']}", "94.74%")

with c2:
    st.metric(f"🌿 {T['classes']}", "38")

with c3:
    st.metric(f"📷 {T['images']}", "54K+")

with c4:
    st.metric(f"📊 {T['total_scans']}", get_total_scans())

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📸 {T['detect']}",
    f"📜 {T['history']}",
    f"🌦 {T['weather']}",
    f"🤖 {T['ask_ai']}",
    f"ℹ️ {T['about']}"
])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"📤 {T['upload']}")

        camera_image = st.camera_input(f"📷 {T['camera']}")

        uploaded_file = st.file_uploader(
            T["choose"],
            type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=T["upload"], use_container_width=True)

        elif camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, caption=T["captured_image"], use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"🔍 {T['result']}")

        if image is None:
            st.info(T["no_upload"])

        else:
            if st.button(f"🌿 {T['detect']}", use_container_width=True):
                img = image.resize(IMG_SIZE)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array, verbose=0)[0]
                top_3_indices = predictions.argsort()[-3:][::-1]

                predicted_index = int(top_3_indices[0])
                confidence = float(predictions[predicted_index])
                predicted_class = class_names[predicted_index]

                crop, disease = split_crop_disease(predicted_class)
                treatment = normalize_treatment(get_treatment(predicted_class))
                severity = get_severity(confidence)

                save_history(crop, disease, confidence)

                st.session_state["last_result"] = {
                    "crop": crop,
                    "disease": disease,
                    "confidence": confidence,
                    "treatment": treatment,
                    "severity": severity,
                    "top_3_indices": top_3_indices.tolist(),
                    "predictions": predictions.tolist()
                }

            if "last_result" in st.session_state:
                result = st.session_state["last_result"]

                crop = result["crop"]
                disease = result["disease"]
                confidence = result["confidence"]
                treatment = result["treatment"]
                severity = result["severity"]
                top_3_indices = result["top_3_indices"]
                predictions = np.array(result["predictions"])

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"## 🌾 {T['crop']}: {crop}")
                st.markdown(f"## 🦠 {T['condition']}: {disease}")
                st.markdown(f"### 📊 {T['confidence']}: {confidence * 100:.2f}%")
                st.progress(confidence)
                st.markdown(f"### {T['severity']}: {severity}")
                show_action_priority(severity, T)
                st.markdown("</div>", unsafe_allow_html=True)

                st.write(f"### 🏆 {T['top3']}")
                for i in top_3_indices:
                    i = int(i)
                    score = float(predictions[i])
                    st.write(f"**{clean_name(class_names[i])}**")
                    st.progress(score)
                    st.caption(f"{score * 100:.2f}%")

                st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
                st.markdown(f"## 💊 {T['action']}")
                st.write(f"**{T['status']}:** {treatment['status']}")
                st.write(f"**{T['cause']}:** {treatment['cause']}")
                st.write(f"**{T['symptoms']}:** {treatment['symptoms']}")
                st.write(f"**{T['spread']}:** {treatment['spread']}")
                st.write(f"**{T['organic']}:** {treatment['organic']}")
                st.write(f"**{T['chemical']}:** {treatment['chemical']}")
                st.write(f"**{T['prevention']}:** {treatment['prevention']}")
                st.write(f"**{T['expert']}:** {treatment['expert']}")
                st.write(f"**{T['risk_level']}:** {treatment['risk']}")
                st.markdown("</div>", unsafe_allow_html=True)

                speak_message = (
                    f"Detected crop is {crop}. "
                    f"Condition is {disease}. "
                    f"Confidence is {confidence * 100:.2f} percent. "
                    f"Severity is {severity}. "
                    f"Organic method: {treatment['organic']}. "
                    f"Chemical method: {treatment['chemical']}."
                )

                if st.button(f"🔊 {T['speak']}", use_container_width=True):
                    speak_text(speak_message)

                pdf_file = create_pdf_report(
                    crop=crop,
                    disease=disease,
                    confidence=confidence,
                    severity=severity,
                    treatment=treatment
                )

                st.download_button(
                    label=f"📄 {T['download']}",
                    data=pdf_file,
                    file_name="krishimitra_ai_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"📜 {T['history']}")

    try:
        history_df = pd.read_csv(HISTORY_PATH)

        if not history_df.empty:
            h1, h2, h3 = st.columns(3)

            with h1:
                st.metric(f"📊 {T['total_scans']}", len(history_df))

            with h2:
                st.metric(f"🦠 {T['unique_diseases']}", history_df["Disease"].nunique())

            with h3:
                st.metric(f"🏆 {T['most_common']}", history_df["Disease"].mode()[0])

            st.write(f"### {T['disease_distribution']}")
            st.bar_chart(history_df["Disease"].value_counts())

        st.write(f"### {T['prediction_history']}")
        st.dataframe(history_df, use_container_width=True)

    except FileNotFoundError:
        st.info(T["no_history"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌦 Live Weather-Based Disease Risk")

    detected_city = get_user_city()

    if detected_city:
        st.success(f"📍 Detected Location: {detected_city}")
        city = detected_city
    else:
        st.warning("Location could not be detected automatically.")
        city = st.text_input("Enter your city", value="Saharanpur")

    if st.button("Get Live Disease Risk", use_container_width=True):
        weather, error = get_live_weather(city)

        if error:
            st.error(error)
        else:
            temperature = weather["temperature"]
            humidity = weather["humidity"]
            condition = weather["condition"]

            risk_level, risk_msg, action = predict_weather_disease_risk(
                temperature, humidity
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("📍 City", weather["city"])

            with c2:
                st.metric("🌡 Temperature", f"{temperature}°C")

            with c3:
                st.metric("💧 Humidity", f"{humidity}%")

            st.info(f"☁ Weather Condition: {condition.title()}")

            if "High" in risk_level:
                st.error(f"{risk_level}: {risk_msg}")
            elif "Medium" in risk_level:
                st.warning(f"{risk_level}: {risk_msg}")
            else:
                st.success(f"{risk_level}: {risk_msg}")

            st.write(f"**Recommended Action:** {action}")

    st.markdown("</div>", unsafe_allow_html=True)


with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"🤖 {T['ask_ai']}")
    st.write(T["ask_desc"])

    question = st.text_input(T["ask_placeholder"])

    if question:
        q = question.lower()

        if "spread" in q or "fail" in q or "फैल" in q:
            st.success(T["ans_spread"])
        elif "water" in q or "watering" in q or "पानी" in q:
            st.success(T["ans_water"])
        elif "fertilizer" in q or "खाद" in q:
            st.success(T["ans_fertilizer"])
        elif "fungicide" in q or "medicine" in q or "दवा" in q:
            st.success(T["ans_fungicide"])
        elif "prevent" in q or "prevention" in q or "बचाव" in q:
            st.success(T["ans_prevention"])
        elif "expert" in q or "doctor" in q or "किसान" in q:
            st.success(T["ans_expert"])
        else:
            st.info(T["ans_default"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown(f"## 🌱 {T['about_title']}")

    st.write(T["about_para"])

    st.markdown(f"### 🌿 {T['features']}")
    st.write(T["features_list"])

    st.markdown(f"### 🧠 {T['technology']}")
    st.write(T["technology_list"])

    st.markdown(f"### 📊 {T['performance']}")
    st.write(T["performance_list"])

    st.markdown(f"### 👨‍💻 {T['made_by']}")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(f"⚠️ {T['warning']}")