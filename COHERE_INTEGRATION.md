# Using Cohere API for Grounded Generation

**Good news**: Cohere works perfectly for biomedical grounding. Actually excellent choice.

---

## Why Cohere Works

✅ **Command-R** (flagship model)
- Strong reasoning capability
- Good instruction following
- Cost-effective
- Supports system prompts

✅ **Command-R Plus** (if you need more capability)
- Better reasoning
- Larger context window (4K tokens)

✅ **Advantages over OpenAI for this task**:
- No API rate limits as strict
- Billing more granular
- Good for research/evaluation
- Command model excellent for grounded tasks

---

## Integration Steps

### 1. Install Cohere SDK

```bash
.\venv\Scripts\pip.exe install cohere
```

Verify:
```bash
.\venv\Scripts\python.exe -c "import cohere; print(cohere.__version__)"
```

---

### 2. Update requirements.txt

Add to `requirements.txt`:

```
cohere>=5.0.0
```

---

### 3. Create Cohere Generator

Create `rag/generator_cohere.py`:

```python
import cohere
import os
from typing import Optional, List

class CohereGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "command-r"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not set")
        
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model
    
    def generate(self, query: str, context: List[str], system_prompt: str = "") -> str:
        """
        Generate answer using Cohere API
        
        Args:
            query: User query
            context: List of evidence chunks
            system_prompt: System instructions
        
        Returns:
            Generated answer
        """
        
        # Format context
        context_text = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        user_prompt = f"""Query: {query}

Evidence:
{context_text}

Please answer the query using ONLY the provided evidence. Cite sources."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt or self._default_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        return response.content[0].text
    
    def _default_system_prompt(self) -> str:
        return """You are a biomedical information assistant. 

Answer ONLY using the provided evidence.
If evidence is insufficient, say so explicitly.
Do not fabricate medical claims.
Cite sources for every claim."""


# Alternative: Simpler streaming version
class CohereGeneratorStreaming:
    def __init__(self, api_key: Optional[str] = None, model: str = "command-r"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.model = model
    
    def generate_stream(self, query: str, context: List[str]):
        """Stream response for real-time output"""
        context_text = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        user_prompt = f"""Query: {query}

Evidence:
{context_text}

Answer concisely using only the evidence."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            stream=True,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        for event in response:
            if hasattr(event, 'delta') and event.delta:
                yield event.delta.message.content[0].text
```

---

### 4. Update Generation Pipeline

Modify `rag/generation_pipeline.py` to support Cohere:

```python
from rag.generator_cohere import CohereGenerator

class GroundedGenerationPipeline:
    def __init__(
        self, 
        vectorstore,
        use_mock=True,
        llm_provider="openai",  # NEW: "openai" or "cohere"
        model=None
    ):
        self.vectorstore = vectorstore
        self.use_mock = use_mock
        self.llm_provider = llm_provider
        
        # Initialize appropriate LLM
        if llm_provider == "cohere":
            self.generator = CohereGenerator(model=model or "command-r")
        elif llm_provider == "openai":
            from rag.generator import GroundedGenerator
            self.generator = GroundedGenerator(use_mock=use_mock, model=model)
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")
        
        # ... rest of init ...
```

---

### 5. Create Cohere Test Script

Create `stabilization_test_cohere.py`:

```python
import os
import sys

os.chdir(r"f:\Users\phili\Documents\Projects\LLM-powered-document-Q&A-system-RAG\rag-qa")
sys.path.insert(0, ".")

from rag.generation_pipeline import GroundedGenerationPipeline

# Make sure API key is set
if not os.getenv("COHERE_API_KEY"):
    print("ERROR: COHERE_API_KEY not set")
    print("Set it: $env:COHERE_API_KEY='your-key-here'")
    sys.exit(1)

# Initialize with Cohere
pipeline = GroundedGenerationPipeline(
    use_mock=False,
    llm_provider="cohere",  # Use Cohere
    model="command-r"
)

# Test queries
test_queries = [
    "What are the main symptoms of diabetes?",
    "What causes Parkinson's disease?",
    "What is leukemia?"
]

print("=" * 80)
print("COHERE LLM GROUNDING VALIDATION TEST")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}/3] Query: {query}")
    print("-" * 80)
    
    try:
        result = pipeline.generate(query)
        
        print(f"Status: {'PASS' if result['valid'] else 'FAIL'}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        print(f"\nAnswer (first 400 chars):")
        print(result['answer'][:400] + "...")
        
        if result['valid']:
            print("\n✓ VALIDATION PASSED")
        else:
            print("\n✗ VALIDATION FAILED")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
```

---

## How to Run

### 1. Set Cohere API Key

**PowerShell (Windows)**:
```powershell
$env:COHERE_API_KEY="your-cohere-key-here"
```

**Bash/Unix**:
```bash
export COHERE_API_KEY="your-cohere-key-here"
```

### 2. Install SDK

```bash
.\venv\Scripts\pip.exe install cohere
```

### 3. Run Test

```bash
.\venv\Scripts\python.exe stabilization_test_cohere.py
```

---

## Cost Comparison

### Cohere Command-R
- Input: $0.50 per million tokens
- Output: $1.50 per million tokens
- **Per query**: ~$0.001-0.003 (very cheap)

### OpenAI gpt-4o-mini
- Input: $0.15 per million tokens
- Output: $0.60 per million tokens
- **Per query**: ~$0.001-0.002 (similar)

**Verdict**: Both affordable for this task.

---

## Advantages of Using Cohere

✅ **Command model is purpose-built for grounding**
✅ **No strict rate limits**
✅ **Good biomedical reasoning**
✅ **Cost-effective**
✅ **API is simpler than OpenAI**
✅ **Streaming built-in**

---

## Integration in Generation Pipeline

To use both providers:

```python
# Use OpenAI
pipeline = GroundedGenerationPipeline(use_mock=False, llm_provider="openai")

# OR use Cohere
pipeline = GroundedGenerationPipeline(use_mock=False, llm_provider="cohere")

# Both support same interface
result = pipeline.generate("What are symptoms of diabetes?")
```

---

## Files to Create

1. **rag/generator_cohere.py** - Cohere integration (new)
2. **stabilization_test_cohere.py** - Test script (new)
3. **requirements.txt** - Add cohere dependency (modify)
4. **rag/generation_pipeline.py** - Support both providers (modify)

---

## Troubleshooting

### Error: "No module named cohere"
```
Solution: .\venv\Scripts\pip.exe install cohere
```

### Error: "COHERE_API_KEY not set"
```
Solution: $env:COHERE_API_KEY="your-key"
```

### Error: "Invalid API key"
```
Solution: Check key at https://dashboard.cohere.com/
```

### Slow Response
```
Solution: Command-R can take 2-5 seconds
This is normal for reasoning models
```

---

## Next Steps

1. Get Cohere API key (free tier available at cohere.com)
2. Set environment variable
3. Install SDK
4. Run stabilization_test_cohere.py
5. If works, proceed with rest of stabilization phase

---

## Recommendation

For **stabilization phase**:

✅ Use **Cohere Command-R** (excellent for grounding)

For **evaluation/production**:
- Compare both (OpenAI vs Cohere)
- Pick based on your results

Both are solid choices for biomedical RAG.

---

**Ready?** Set your COHERE_API_KEY and run the test!
