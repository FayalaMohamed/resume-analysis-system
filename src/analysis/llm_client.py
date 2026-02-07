"""LLM client for resume improvement suggestions.

Supports both local (Ollama) and API-based (OpenRouter) models
with automatic fallback.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src to path for importing Config
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils import Config
except ImportError:
    Config = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    success: bool
    error: Optional[str] = None


class LLMClient:
    """Unified LLM client with fallback support."""
    
    # OpenRouter free tier models
    OPENROUTER_MODELS = [
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free", 
        "openai/gpt-oss-20b:free",
        "deepseek/deepseek-r1-0528:free",
    ]
    
    # Ollama local models
    OLLAMA_MODELS = [
        "ministral:3b-instruct-2512-q4_K_M",
    ]
    
    def __init__(
        self,
        primary: str = "openrouter",
        fallback: str = "ollama",
        api_key: Optional[str] = None
    ):
        """Initialize LLM client.
        
        Args:
            primary: Primary provider ('openrouter' or 'ollama')
            fallback: Fallback provider
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env var or .env file)
        """
        self.primary = primary
        self.fallback = fallback
        
        # Get API key from: 1) parameter, 2) Config class (from .env), 3) environment variable
        if api_key:
            self.api_key = api_key
        elif Config and hasattr(Config, 'OPENROUTER_API_KEY') and Config.OPENROUTER_API_KEY:
            self.api_key = Config.OPENROUTER_API_KEY
        else:
            self.api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        self.openrouter_available = self._check_openrouter()
        self.ollama_available = self._check_ollama()
    
    def _check_openrouter(self) -> bool:
        """Check if OpenRouter is available."""
        if not self.api_key:
            return False
        
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available locally."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> LLMResponse:
        """Generate text using LLM with fallback.
        
        Args:
            prompt: Input prompt
            temperature: Creativity (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with generated text
        """
        # Try primary first
        if self.primary == "openrouter" and self.openrouter_available:
            result = self._call_openrouter(prompt, temperature, max_tokens)
            if result.success:
                return result
        
        if self.primary == "ollama" and self.ollama_available:
            result = self._call_ollama(prompt, temperature, max_tokens)
            if result.success:
                return result
        
        # Try fallback
        if self.fallback == "openrouter" and self.openrouter_available:
            return self._call_openrouter(prompt, temperature, max_tokens)
        
        if self.fallback == "ollama" and self.ollama_available:
            return self._call_ollama(prompt, temperature, max_tokens)
        
        # Neither available
        return LLMResponse(
            text="",
            model="none",
            success=False,
            error="No LLM provider available. Please set OPENROUTER_API_KEY or install Ollama."
        )
    
    def _call_openrouter(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call OpenRouter API."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://ats-resume-analyzer.local",
                "X-Title": "ATS Resume Analyzer",
                "Content-Type": "application/json"
            }
            
            # Try models in order
            for model in self.OPENROUTER_MODELS:
                try:
                    data = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result["choices"][0]["message"]["content"]
                        return LLMResponse(
                            text=text,
                            model=model,
                            success=True
                        )
                except Exception as e:
                    continue
            
            return LLMResponse(
                text="",
                model="openrouter",
                success=False,
                error="All OpenRouter models failed"
            )
            
        except Exception as e:
            return LLMResponse(
                text="",
                model="openrouter",
                success=False,
                error=str(e)
            )
    
    def _call_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Call local Ollama instance."""
        try:
            import requests
            
            for model in self.OLLAMA_MODELS:
                try:
                    data = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        }
                    }
                    
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json=data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return LLMResponse(
                            text=result["response"],
                            model=f"ollama/{model}",
                            success=True
                        )
                except Exception:
                    continue
            
            return LLMResponse(
                text="",
                model="ollama",
                success=False,
                error="All Ollama models failed"
            )
            
        except Exception as e:
            return LLMResponse(
                text="",
                model="ollama",
                success=False,
                error=str(e)
            )
    
    def improve_bullet_point(self, bullet: str, context: str = "") -> LLMResponse:
        """Improve a resume bullet point.
        
        Args:
            bullet: Original bullet text
            context: Additional context (job role, company, etc.)
            
        Returns:
            LLMResponse with improved bullet
        """
        prompt = f"""Improve this resume bullet point to be more impactful and ATS-friendly.

Original: {bullet}

Guidelines:
- Start with a strong action verb
- Include specific metrics and quantifiable achievements
- Be concise (15-25 words)
- Focus on results and impact, not just responsibilities
{context}

Provide ONLY the improved bullet point, nothing else:"""

        return self.generate(prompt, temperature=0.5, max_tokens=100)
    
    def suggest_keywords(self, resume_text: str, job_description: str = "") -> LLMResponse:
        """Suggest keywords to add to resume.
        
        Args:
            resume_text: Current resume text
            job_description: Optional job description
            
        Returns:
            LLMResponse with keyword suggestions
        """
        jd_section = ""
        if job_description:
            jd_section = f"Job Description:\n{job_description[:500]}\n\n"
        
        prompt = f"""Analyze this resume and suggest 5-10 important keywords or skills that should be added.

Resume:
{resume_text[:1000]}

{jd_section}Provide a comma-separated list of keywords that would strengthen this resume for ATS systems:"""

        return self.generate(prompt, temperature=0.3, max_tokens=150)
    
    def rewrite_section(self, section_text: str, section_type: str) -> LLMResponse:
        """Rewrite a resume section for better impact.
        
        Args:
            section_text: Current section text
            section_type: Type of section (Experience, Skills, etc.)
            
        Returns:
            LLMResponse with rewritten section
        """
        prompt = f"""Rewrite this {section_type} section to be more professional and ATS-optimized.

Original:
{section_text}

Guidelines:
- Use strong action verbs
- Include metrics where possible
- Be concise but detailed
- Use industry-standard terminology

Rewritten version:"""

        return self.generate(prompt, temperature=0.6, max_tokens=400)
    
    def enhance_bullet_critique(self, bullet: str) -> LLMResponse:
        """Critique and suggest enhancements for a good bullet point.
        
        Args:
            bullet: Current bullet text (already considered good)
            
        Returns:
            LLMResponse with constructive critique and enhancement suggestion
        """
        prompt = f"""Critique this resume bullet point and suggest specific enhancements to make it even stronger.

Current bullet: "{bullet}"

Provide a brief critique (1 sentence) followed by an enhanced version.

Guidelines for critique:
- Could it benefit from stronger/more specific metrics?
- Is the impact clear and quantified?
- Could the action verb be more powerful or specific?
- Is there a clearer way to express the achievement?
- Does it lead with results rather than activities?

Format your response as:
💭 Critique: [your constructive feedback]
✨ Enhanced: [the improved version]

Keep the enhanced version concise (15-25 words) and impactful:"""

        return self.generate(prompt, temperature=0.6, max_tokens=150)
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM provider status.
        
        Returns:
            Dictionary with provider availability
        """
        return {
            "openrouter": {
                "available": self.openrouter_available,
                "models": self.OPENROUTER_MODELS if self.openrouter_available else [],
            },
            "ollama": {
                "available": self.ollama_available,
                "models": self.OLLAMA_MODELS if self.ollama_available else [],
            },
            "primary": self.primary,
            "fallback": self.fallback,
        }
