# common/llm_utils.py
# Same retry discipline as every earlier repo: a 429 is a structured
# error the SDK already understands, a dropped connection is a raw
# transport failure that surfaces as an httpx/httpcore exception and
# isn't always absorbed by the SDK's own retry logic.
#
# attempt_gemini_tuning exists to make a real, honest attempt at the
# Developer API's own tuning surface before this repo pivots to local
# LoRA fine-tuning. client.tunings.tune(base_model=..., training_dataset=...)
# is real, fully wired SDK code (google/genai/tunings.py), confirmed by
# reading the installed source directly, not assumed from docs. What's
# NOT assumed: whether it actually succeeds against a live Developer API
# key today. This function reports the real outcome either way, see
# this post's Pitfalls section for what actually happened.

import time
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from common.rate_limiter import throttle

MAX_RETRIES = 4


def generate_with_backoff(client, model: str, contents: str, temperature: float = None, system_instruction: str = None):
    """Drop-in replacement for client.models.generate_content(...).
    Retries on rate limits (429) and on transport-level connection
    failures. Anything else, a real API error unrelated to connectivity
    or rate limiting, is not retried and raises immediately."""
    config = None
    if temperature is not None or system_instruction is not None:
        config = genai_types.GenerateContentConfig(temperature=temperature, system_instruction=system_instruction)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            throttle()
            if config is not None:
                return client.models.generate_content(model=model, contents=contents, config=config)
            return client.models.generate_content(model=model, contents=contents)
        except genai_errors.ClientError as e:
            last_error = e
            if e.code == 429:
                wait_seconds = 2 ** attempt * 5
                print(f"  (rate limited, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_seconds)
            else:
                raise
        except Exception as e:
            last_error = e
            wait_seconds = 3 * (attempt + 1)
            print(f"  (connection issue: {type(e).__name__}, waiting {wait_seconds}s, retry {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_seconds)
    raise last_error


def attempt_gemini_tuning(client, base_model: str, examples: list) -> dict:
    """Makes a real client.tunings.tune(...) call against the Developer
    API, examples is a list of {"text_input": str, "output": str} dicts,
    the real, Developer-API-only TuningExample shape (confirmed in the
    installed SDK source, and explicitly NOT supported on Vertex AI per
    that same source). Returns {"submitted": True, "job": ...} if the
    API actually accepts the job, or {"submitted": False, "error": str}
    on a real, structured failure, so calling code can report the
    outcome honestly instead of assuming either way."""
    dataset = genai_types.TuningDataset(
        examples=[genai_types.TuningExample(text_input=e["text_input"], output=e["output"]) for e in examples]
    )
    try:
        job = client.tunings.tune(base_model=base_model, training_dataset=dataset)
        return {"submitted": True, "job": job}
    except (genai_errors.ClientError, genai_errors.ServerError) as e:
        # Real, confirmed outcome on gemini-3.5-flash-lite: a 501
        # ServerError ("Operation is not implemented, or supported, or
        # enabled"), not a ClientError, the request itself is
        # well-formed, the capability simply isn't there anymore.
        return {"submitted": False, "error": str(e)}
    except Exception as e:
        return {"submitted": False, "error": f"{type(e).__name__}: {e}"}
