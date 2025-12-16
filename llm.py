from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
import time
import random
from time import sleep

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English",
                min_interval_sec: float = 13.0):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

        # Throttling state (per process)
        self._min_interval_sec = float(min_interval_sec)
        self._last_call_ts = 0.0
    
    def _throttle(self) -> None:
        """ Ensure we don't exceed RPM by enforcing a minimum interval between calls."""
        now = time.time()
        wait = self._min_interval_sec - (now - self._last_call_ts)
        if wait > 0:
            sleep(wait)
        self._last_call_ts = time.time()

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Best-effort check for 429 / rate-limit errors across providers."""
        msg = str(e).lower()
        return ("ratelimiterror" in msg)


    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            # ✅ Always throttle BEFORE calling the API
            self._throttle()

            max_retries = 5
            backoff = 33  # seconds; you said you don't care about latency
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(
                        messages=messages,
                        temperature=0,
                        model=self.model,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")

                    if attempt == max_retries - 1:
                        raise

                    # Only do long backoff for rate-limit errors
                    if self._is_rate_limit_error(e):
                        # Add a bit of jitter to avoid thundering herd / boundary effects
                        sleep(backoff + random.uniform(0, 3))
                        backoff = min(backoff * 2, 120)
                    else:
                        # For non-rate-limit errors, a short wait is usually enough (or you can raise)
                        sleep(3)

                    # Throttle again before the next retry attempt
                    self._throttle()

        else:
            response = self.llm.create_chat_completion(messages=messages, temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
