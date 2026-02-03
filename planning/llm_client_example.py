# Dual LLM Configuration Example
# This file shows how to configure both Ollama (local) and OpenRouter (API) options
#
# Available FREE OpenRouter Models:
# - openai/gpt-oss-120b:free           (120B parameters, largest)
# - openai/gpt-oss-20b:free            (20B parameters, balanced)
# - z-ai/glm-4.5-air:free              (GLM model)
# - deepseek/deepseek-r1-0528:free     (DeepSeek reasoning model)
# - mistralai/mistral-small-3.1-24b-instruct:free  (24B, good default)
# - meta-llama/llama-3.3-70b-instruct:free         (70B Llama)
#
# Default used below: mistralai/mistral-small-3.1-24b-instruct:free

import os
import requests
from typing import Optional, Dict, Any


class LLMClient:
    """
    Dual LLM client supporting both local (Ollama) and API (OpenRouter) options.
    
    Usage:
        # Try OpenRouter first, fallback to Ollama
        client = LLMClient(primary='openrouter', fallback='ollama')
        
        # Or use only local
        client = LLMClient(primary='ollama')
        
        response = client.generate("Rewrite this bullet point: ...")
    """
    
    def __init__(self, 
                 primary: str = 'openrouter',
                 fallback: Optional[str] = 'ollama',
                 openrouter_model: str = 'mistralai/mistral-small-3.1-24b-instruct:free',  # Or choose from: openai/gpt-oss-120b:free, openai/gpt-oss-20b:free, z-ai/glm-4.5-air:free, deepseek/deepseek-r1-0528:free, meta-llama/llama-3.3-70b-instruct:free
                 openrouter_api_key: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            primary: Primary provider ('openrouter' or 'ollama')
            fallback: Fallback provider if primary fails (None for no fallback)
            openrouter_model: Model identifier from OpenRouter
            openrouter_api_key: API key (reads from OPENROUTER_API_KEY env var if not provided)
        """
        self.primary = primary
        self.fallback = fallback
        self.openrouter_model = openrouter_model
        
        # Get API key from parameter or environment
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        
        if primary == 'openrouter' and not self.openrouter_api_key:
            print("⚠️ Warning: OPENROUTER_API_KEY not set. OpenRouter will fail.")
        
        # Ollama settings
        self.ollama_model = 'ministral:3b-instruct-2512-q4_K_M'
        self.ollama_url = 'http://localhost:11434/api/generate'
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate text using primary provider, fallback on failure.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 - 1.0)
            
        Returns:
            Generated text
        """
        # Try primary first
        try:
            if self.primary == 'openrouter':
                return self._generate_openrouter(prompt, temperature)
            else:
                return self._generate_ollama(prompt, temperature)
        except Exception as e:
            print(f"⚠️ Primary provider ({self.primary}) failed: {e}")
            
            # Try fallback if configured
            if self.fallback:
                print(f"🔄 Falling back to {self.fallback}...")
                try:
                    if self.fallback == 'openrouter':
                        return self._generate_openrouter(prompt, temperature)
                    else:
                        return self._generate_ollama(prompt, temperature)
                except Exception as fallback_error:
                    raise Exception(f"Both providers failed. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise
    
    def _generate_openrouter(self, prompt: str, temperature: float) -> str:
        """Generate using OpenRouter API."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.openrouter_model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant for improving resumes.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
    
    def _generate_ollama(self, prompt: str, temperature: float) -> str:
        """Generate using local Ollama."""
        data = {
            'model': self.ollama_model,
            'prompt': prompt,
            'temperature': temperature,
            'stream': False
        }
        
        response = requests.post(
            self.ollama_url,
            json=data,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['response']
    
    def test_providers(self) -> Dict[str, bool]:
        """Test both providers and return availability status."""
        results = {}
        
        # Test Ollama
        try:
            self._generate_ollama("Test", temperature=0.1)
            results['ollama'] = True
            print("✅ Ollama is available")
        except Exception as e:
            results['ollama'] = False
            print(f"❌ Ollama unavailable: {e}")
        
        # Test OpenRouter
        if self.openrouter_api_key:
            try:
                self._generate_openrouter("Test", temperature=0.1)
                results['openrouter'] = True
                print("✅ OpenRouter is available")
            except Exception as e:
                results['openrouter'] = False
                print(f"❌ OpenRouter unavailable: {e}")
        else:
            results['openrouter'] = False
            print("⚠️ OpenRouter API key not set")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize with both options
    client = LLMClient(
        primary='openrouter',  # Try API first
        fallback='ollama'      # Fall back to local if API fails
    )
    
    # Test which providers are available
    print("\n🔍 Testing LLM providers...")
    availability = client.test_providers()
    
    # Example: Improve a resume bullet point
    if any(availability.values()):
        prompt = """
        Rewrite this resume bullet point to be more impactful:
        
        Original: "Helped with website development"
        
        Make it:
        - Start with a strong action verb
        - Include specific metrics if possible
        - Be concise (1 line)
        
        Improved version:
        """
        
        print("\n✍️ Testing generation...")
        try:
            result = client.generate(prompt)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n❌ No LLM providers available. Please configure Ollama or OpenRouter.")
