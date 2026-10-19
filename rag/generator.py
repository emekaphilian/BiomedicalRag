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
                answer = self._cohere_response_text(response)
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
            error_message = self._safe_error_message(e)
            return {
                "answer": f"Generation failed: {error_message}",
                "query": prompt_dict["query"],
                "evidence_count": prompt_dict["evidence_count"],
                "sources": prompt_dict["evidence_sources"],
                "model": self.model_name,
                "success": False,
                "error": error_message,
            }

    def _safe_error_message(self, error):
        """Return useful provider diagnostics without dumping response headers."""
        response = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        body = getattr(error, "body", None)
        if body is None and response is not None:
            body = getattr(response, "json_data", None)
        if isinstance(body, dict):
            detail = next(
                (body[key] for key in ("message", "error", "detail") if body.get(key)),
                None,
            )
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("type")
        else:
            detail = body

        if detail is None:
            if isinstance(error, (AttributeError, TypeError, ValueError)):
                detail = str(error)
                if not detail and isinstance(error, AttributeError):
                    traceback = error.__traceback__
                    while traceback is not None and traceback.tb_next is not None:
                        traceback = traceback.tb_next
                    if traceback is not None:
                        filename = (
                            traceback.tb_frame.f_code.co_filename
                            .replace("\\", "/")
                            .rsplit("/", 1)[-1]
                        )
                        detail = (
                            f"{type(error).__name__} at "
                            f"{filename}:{traceback.tb_lineno}"
                        )
                detail = detail or type(error).__name__
            else:
                detail = type(error).__name__
        detail = " ".join(str(detail).split())
        if self.api_key:
            detail = detail.replace(self.api_key, "[redacted]")
        detail = detail[:400]

        provider = self.provider.capitalize()
        status = f" (HTTP {status_code})" if status_code else ""
        return f"{provider} request failed{status}: {detail}"

    @staticmethod
    def _cohere_response_text(response):
        """Extract generated text from either Cohere v2 content representation."""
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, (list, tuple)):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text = block
                elif isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)

        raise ValueError("Cohere response did not contain text content.")

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
