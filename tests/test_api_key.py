"""Test script to check if API key is being loaded."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test 1: Check if .env file exists
env_path = Path(__file__).parent / ".env"
print(f".env file exists: {env_path.exists()}")

if env_path.exists():
    with open(env_path) as f:
        content = f.read()
        has_key = "OPENROUTER_API_KEY" in content and "your_" not in content
        print(f".env has API key set: {has_key}")

# Test 2: Check if dotenv is installed
try:
    from dotenv import load_dotenv
    print("python-dotenv is installed: YES")
except ImportError:
    print("python-dotenv is installed: NO")

# Test 3: Check Config class
try:
    from utils import Config
    print(f"Config.OPENROUTER_API_KEY: {'Set' if Config.OPENROUTER_API_KEY else 'Not set'}")
    print(f"Config.OPENROUTER_API_KEY value: {Config.OPENROUTER_API_KEY[:20]}..." if Config.OPENROUTER_API_KEY else "Empty")
except Exception as e:
    print(f"Error importing Config: {e}")

# Test 4: Check LLMClient
try:
    from analysis import LLMClient
    llm = LLMClient()
    print(f"LLMClient api_key set: {'Yes' if llm.api_key else 'No'}")
    print(f"OpenRouter available: {llm.openrouter_available}")
except Exception as e:
    print(f"Error with LLMClient: {e}")

print("\n" + "="*50)
print("To fix:")
print("1. Make sure .env file exists (cp .env.example .env)")
print("2. Add your OpenRouter API key to .env file")
print("3. Install python-dotenv: pip install python-dotenv")
print("4. Restart the Streamlit app")
