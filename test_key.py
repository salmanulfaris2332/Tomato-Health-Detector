import google.generativeai as genai

# PASTE YOUR KEY HERE
GOOGLE_API_KEY = "AIzaSyBsVEalTK-Zdw_3OfxJSWdTVCunO2K0few"

genai.configure(api_key=GOOGLE_API_KEY)

print("Checking for available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ AVAILABLE: {m.name}")
    print("\nSuccess! Your API Key is working.")
except Exception as e:
    print(f"\n❌ ERROR: Your API Key or Account has a problem:\n{e}")