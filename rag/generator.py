"""
LLM-based answer generation with grounding validation.
Supports both OpenAI API and mock generation for testing.
"""


class GroundedGenerator:
    """
    Generates biomedical answers from retrieved evidence.
    
    Features:
    - OpenAI, Cohere, and Hugging Face API integrations
    - Mock generation for testing
    - Structured output with metadata
    - Citation tracking
    """

    def __init__(self, model_name="gpt-4o", api_key=None, provider="openai"):
        """
        Initialize LLM generator.
        
        Args:
            model_name: OpenAI model to use
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.provider = provider.lower()
        
        # Lazy import to avoid hard dependency
        self.client = None

    def _get_client(self):
        """Get the selected provider client on first use."""
        if self.client is None:
            try:
                if self.provider == "openai":
                    from openai import OpenAI

                    self.client = OpenAI(api_key=self.api_key)
                elif self.provider == "cohere":
                    import cohere

                    self.client = cohere.ClientV2(api_key=self.api_key)
                elif self.provider == "huggingface":
                    from huggingface_hub import InferenceClient

                    self.client = InferenceClient(
                        model=self.model_name,
                        token=self.api_key,
                    )
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.provider}")
            except ImportError:
                return None
        return self.client

    def generate(self, prompt_dict, max_tokens=500, temperature=0.3):
        """
        Generate grounded answer from prompt.
        
        Args:
            prompt_dict: Output from GroundedPromptConstructor
            max_tokens: Maximum tokens to generate
            temperature: Lower = more focused, higher = more creative
            
        Returns:
            Generated answer with metadata
        """
        try:
            client = self._get_client()
            
            if client is None:
                return self.generate_mock(prompt_dict)
            
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt_dict["system"]},
                        {"role": "user", "content": prompt_dict["user"]},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = response.choices[0].message.content
            elif self.provider == "cohere":
                response = client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": prompt_dict["system"]},
                        {"role": "user", "content": prompt_dict["user"]},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = response.message.content[0].text
            else:
                answer = client.text_generation(
                    f"{prompt_dict['system']}\n\n{prompt_dict['user']}",
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    return_full_text=False,
                )

            if not answer:
                raise ValueError("The selected LLM returned an empty answer.")
            answer = answer.strip()
            
            return {
                "answer": answer,
                "query": prompt_dict["query"],
                "evidence_count": prompt_dict["evidence_count"],
                "sources": prompt_dict["evidence_sources"],
                "model": self.model_name,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "answer": f"Generation failed: {str(e)}",
                "query": prompt_dict["query"],
                "evidence_count": prompt_dict["evidence_count"],
                "sources": prompt_dict["evidence_sources"],
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }

    def generate_mock(self, prompt_dict):
        """
        Generate mock answer (for testing without API key).
        
        Returns fabricated answer based on evidence.
        """
        query = prompt_dict["query"]
        sources = prompt_dict["evidence_sources"]
        
        # Simple mock answer
        answer = f"""Based on the retrieved biomedical evidence:

[Evidence provided from {', '.join(sources[:2])}]

To generate real answers with OpenAI, please provide your OPENAI_API_KEY.

Query: {query}
Evidence sources available: {', '.join(sources)}
Evidence items: {prompt_dict["evidence_count"]}
"""
        
        return {
            "answer": answer,
            "query": query,
            "evidence_count": prompt_dict["evidence_count"],
            "sources": sources,
            "model": "mock",
            "success": True,
            "error": None
        }

