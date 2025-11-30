# app/llm_agent.py
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Try to import the Google GenAI SDK in a way that works across versions.
try:
    # modern import style
    from google import genai
except Exception:
    try:
        # older packaging fallback
        import genai
    except Exception:
        genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if genai is None:
    raise ImportError(
        "Google GenAI SDK (google-genai / genai) not installed. "
        "Install with: pip install google-genai"
    )

# Initialize client.
# Some SDK versions accept api_key in constructor, some pick from env var.
try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
except Exception:
    # fallback: try environment-key-only client
    client = genai.Client()

def _assemble_prompt(summary: dict, imbalance: Optional[dict] = None) -> str:
    prompt_lines = [
        "You are an expert data scientist. Produce a clear, structured EDA report for the dataset below.",
        "Include: (1) Key observations (shape, memory, missing, types), (2) Issues to fix, (3) Recommended preprocessing steps and why,",
        "(4) Imbalance handling suggestions if relevant, (5) Quick feature-engineering ideas and next steps for modeling.",
        "",
        "DATASET_SUMMARY:",
        str(summary),
    ]
    if imbalance:
        prompt_lines += ["", "IMBALANCE_INFO:", str(imbalance)]
    return "\n".join(prompt_lines)

def generate_report(summary: dict, imbalance: Optional[dict] = None, max_tokens: int = 1200) -> str:
    """
    Generate a human-readable EDA report using Gemini.
    This function attempts SDK call patterns that vary across genai versions.
    """
    prompt = _assemble_prompt(summary, imbalance)

    # Try multiple SDK call styles to be robust across versions
    # 1) Newer style: client.generate() or client.models.generate_content
    try:
        # Many newer snippets use client.generate_text or client.models.generate_content
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                temperature=0.0,
                max_output_tokens=max_tokens,
            )
            # resp may have .text or .content or nested structure
            text = getattr(resp, "text", None) or getattr(resp, "content", None)
            if isinstance(text, (list, tuple)):
                # join text bits
                text = "\n".join(getattr(item, "text", str(item)) for item in text)
            return str(text)
    except Exception:
        pass

    # 2) Alternate style: client.generate_text(...)
    try:
        if hasattr(client, "generate_text"):
            resp = client.generate_text(model=GEMINI_MODEL, text=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
    except Exception:
        pass

    # 3) Last resort: client.create() pattern
    try:
        if hasattr(client, "create"):
            resp = client.create(model=GEMINI_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.0)
            # try common response fields
            text = getattr(resp, "text", None) or getattr(resp, "choices", None)
            if isinstance(text, list):
                text = "\n".join([getattr(c, "text", str(c)) for c in text])
            return str(text)
    except Exception as e:
        # If everything fails, return a fallback message (so API still works offline)
        return (
            "LLM generation failed. Please check your google-genai SDK and GEMINI_API_KEY.\n"
            f"SDK error: {e}\n\n"
            "Fallback: Provide the following summary to a human analyst:\n"
            + str(summary)
        )
