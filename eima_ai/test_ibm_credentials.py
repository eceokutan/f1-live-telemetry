"""
IBM Credentials Test Script

Tests each IBM service credential independently to identify connection issues.
"""

import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()


def test_watsonx_llm():
    """Test WatsonX LLM (Granite) credentials."""
    print("\n" + "=" * 60)
    print("Testing WatsonX LLM (Granite)")
    print("=" * 60)

    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    print(f"URL: {url}")
    print(f"Project ID: {project_id[:8]}..." if project_id else "Project ID: NOT SET")
    print(f"API Key: {api_key[:8]}..." if api_key else "API Key: NOT SET")

    if not api_key or not project_id:
        print("❌ FAIL: Missing credentials")
        return False

    # Test 1: Check if langchain_ibm is installed
    print("\n[1] Checking langchain_ibm installation...")
    try:
        from langchain_ibm import WatsonxLLM
        print("✅ langchain_ibm is installed")
    except ImportError as e:
        print(f"❌ FAIL: langchain_ibm not installed: {e}")
        print("   Install with: pip install langchain-ibm")
        return False

    # Test 2: Check ibm_watsonx_ai SDK
    print("\n[2] Checking ibm_watsonx_ai SDK...")
    try:
        import ibm_watsonx_ai
        print(f"✅ ibm_watsonx_ai version: {ibm_watsonx_ai.__version__}")
    except ImportError as e:
        print(f"❌ FAIL: ibm_watsonx_ai not installed: {e}")
        print("   Install with: pip install ibm-watsonx-ai")
        return False

    # Test 3: Try to initialize the LLM
    print("\n[3] Initializing WatsonxLLM...")
    try:
        llm = WatsonxLLM(
            model_id="ibm/granite-3-8b-instruct",
            url=url,
            project_id=project_id,
            apikey=api_key,
            params={
                "max_new_tokens": 50,
                "temperature": 0.7,
                "decoding_method": "greedy",
            },
        )
        print("✅ WatsonxLLM initialized successfully")
    except Exception as e:
        print(f"❌ FAIL: Could not initialize WatsonxLLM: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

    # Test 4: Make a test API call
    print("\n[4] Making test API call...")
    try:
        response = llm.invoke("Say 'Hello' in one word.")
        print(f"✅ API call successful!")
        print(f"   Response: {response[:100]}..." if len(response) > 100 else f"   Response: {response}")
        return True
    except Exception as e:
        print(f"❌ FAIL: API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Try to get more details
        if hasattr(e, 'response'):
            print(f"   Response status: {getattr(e.response, 'status_code', 'N/A')}")
            try:
                print(f"   Response body: {e.response.text[:500]}")
            except:
                pass
        return False


def test_watson_tts():
    """Test Watson Text-to-Speech credentials."""
    print("\n" + "=" * 60)
    print("Testing Watson Text-to-Speech")
    print("=" * 60)

    api_key = os.getenv("WATSON_TTS_API_KEY")
    url = os.getenv("WATSON_TTS_URL")

    print(f"URL: {url}")
    print(f"API Key: {api_key[:8]}..." if api_key else "API Key: NOT SET")

    if not api_key or not url:
        print("❌ FAIL: Missing credentials")
        return False

    # Test with aiohttp (same as the app uses)
    print("\n[1] Testing API connection...")
    try:
        import requests
        from requests.auth import HTTPBasicAuth

        # Get voices list (lightweight test)
        test_url = f"{url}/v1/voices"
        response = requests.get(
            test_url,
            auth=HTTPBasicAuth("apikey", api_key),
            timeout=10
        )

        if response.status_code == 200:
            voices = response.json().get("voices", [])
            print(f"✅ TTS API connection successful!")
            print(f"   Available voices: {len(voices)}")
            return True
        else:
            print(f"❌ FAIL: API returned status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_watson_stt():
    """Test Watson Speech-to-Text credentials."""
    print("\n" + "=" * 60)
    print("Testing Watson Speech-to-Text")
    print("=" * 60)

    api_key = os.getenv("WATSON_STT_API_KEY")
    url = os.getenv("WATSON_STT_URL")

    print(f"URL: {url}")
    print(f"API Key: {api_key[:8]}..." if api_key else "API Key: NOT SET")

    if not api_key or not url:
        print("❌ FAIL: Missing credentials")
        return False

    # Test with requests
    print("\n[1] Testing API connection...")
    try:
        import requests
        from requests.auth import HTTPBasicAuth

        # Get models list (lightweight test)
        test_url = f"{url}/v1/models"
        response = requests.get(
            test_url,
            auth=HTTPBasicAuth("apikey", api_key),
            timeout=10
        )

        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ STT API connection successful!")
            print(f"   Available models: {len(models)}")
            return True
        else:
            print(f"❌ FAIL: API returned status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ FAIL: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def main():
    print("IBM Credentials Diagnostic Test")
    print("================================\n")

    results = {}

    # Test all services
    results["WatsonX LLM"] = test_watsonx_llm()
    results["Watson TTS"] = test_watson_tts()
    results["Watson STT"] = test_watson_stt()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {service}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All credentials are working correctly!")
    else:
        print("Some credentials failed. Check the detailed output above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
