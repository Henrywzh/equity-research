# Groq 2026 模型字典
GROQ_MODELS = {
    # --- 第一层：高频过滤器 (Fast Tier) ---
    "scraper": {
        "model_id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B Instant",
        "rpd": 14400,
        "rpm": 30,
        "description": "极速提取，每日 1.4w 次配额。适合初步清洗网页和去重。",
        "suggested_sleep": 2.0  # 建议每次调用后休眠秒数
    },

    # --- 第二层：主力分析器 (Main Tier) ---
    "analyst": {
        "model_id": "llama-4-scout-17b",
        "name": "Llama 4 Scout (2026 New)",
        "rpd": 1000,
        "rpm": 30,
        "description": "2026 最新模型，逻辑极佳。适合岗位分类、总结亮点。",
        "suggested_sleep": 2.0
    },

    # --- 第三层：深度推理/大模型 (Premium Tier) ---
    "expert": {
        "model_id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B Versatile",
        "rpd": 1000,
        "rpm": 30,
        "description": "参数量大，理解力最强。适合简历打分、面试策略生成。",
        "suggested_sleep": 2.0
    },

    # --- 特种兵：多语言/中文支持 (Region Specific) ---
    "chinese_specialist": {
        "model_id": "qwen/qwen3-32b",
        "name": "Qwen 3 32B",
        "rpd": 1000,
        "rpm": 60,
        "description": "阿里系模型，处理中文岗位信息、国内大厂招聘的首选。",
        "suggested_sleep": 1.0
    },

    # --- 安全卫士 (Guardrail) ---
    "security": {
        "model_id": "meta-llama/llama-guard-4-12b",
        "name": "Llama Guard 4",
        "rpd": 14400,
        "rpm": 30,
        "description": "自动检测 JD 是否为虚假信息或诈骗链接。",
        "suggested_sleep": 2.0
    }
}


# 辅助函数：快速获取模型 ID
def get_model(role: str) -> str:
    """
    'scraper' -> instant
    'analyst' -> main
    'expert' -> expert
    'chinese_specialist' -> qwen
    """
    return GROQ_MODELS.get(role, GROQ_MODELS["analyst"])["model_id"]


# ---------------------------------------------------------------------------
# MODELS — Comprehensive Groq model registry for this project
# ---------------------------------------------------------------------------
MODELS = {
    # --- THE HEAVYWEIGHT (Best for reasoning and nuance) ---
    "llama_3_3_70b": {
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "name": "Llama 3.3 70B (Versatile)",
        "rpd": 1000,
        "rpm": 30,
        "tpm": 12000,
        "supports_vision": False,
        "supports_thinking": True,
        "description": "GPT-4 level intelligence. Best for complex reasoning and high-quality writing.",
        "suggested_sleep": 2.1
    },
    # --- THE LOGICIAN (Best for coding, math, and structured data) ---
    "qwen_3_32b": {
        "model_id": "qwen/qwen3-32b",
        "provider": "groq",
        "name": "Qwen 3 32B (Logic/Coding)",
        "rpd": 1000,
        "rpm": 60,
        "tpm": 6000,
        "supports_vision": False,
        "supports_thinking": False,
        "description": "Alibaba's top-tier logic model. Exceptional at Python and math tasks.",
        "suggested_sleep": 1.1
    },
    # --- THE MULTIMODAL SCOUT (Best for vision/files) ---
    "llama_4_scout": {
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "name": "Llama 4 Scout 17B (Vision)",
        "rpd": 1000,
        "rpm": 30,
        "tpm": 30000,
        "supports_vision": True,
        "supports_thinking": False,
        "description": "Next-gen efficient model. Native vision support and high token throughput.",
        "suggested_sleep": 2.0
    },
    # --- THE SPEED DEMON (Best for high-volume simple tasks) ---
    "llama_3_1_8b": {
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "name": "Llama 3.1 8B (High Volume)",
        "rpd": 14400,
        "rpm": 30,
        "tpm": 6000,
        "supports_vision": False,
        "supports_thinking": False,
        "description": "Massive daily limit. Perfect for classification, summarization, and fast chat.",
        "suggested_sleep": 0.5
    },
    # --- THE MASSIVE OPEN SOURCE (Experimental) ---
    "gpt_oss_120b": {
        "model_id": "openai/gpt-oss-120b",
        "provider": "groq",
        "name": "GPT-OSS 120B",
        "rpd": 1000,
        "rpm": 30,
        "tpm": 8000,
        "supports_vision": False,
        "supports_thinking": True,
        "description": "Massive parameter count. High intelligence but lower token-per-minute limits.",
        "suggested_sleep": 2.5
    }
}


# ---------------------------------------------------------------------------
# MARITIME_ANALYST_MODELS — Models approved for ship detection QA
# Groq entries reference MODELS above to avoid duplication.
# ---------------------------------------------------------------------------
MARITIME_ANALYST_MODELS = {
    # ── Groq (free-tier, via GROQ_API_KEY) ───────────────────────────────
    # Primary vision QA model: sees the annotated map, runs STEP A
    "llama_4_scout": MODELS["llama_4_scout"],
    # Best reasoning model: no vision, runs STEP B+C on CSV trend data only
    "llama_3_3_70b": MODELS["llama_3_3_70b"],
    # High-volume fallback: auto-selected when Groq RPD quota is near exhaustion
    "llama_3_1_8b":  MODELS["llama_3_1_8b"],

    # ── OpenRouter (via OPENROUTER_API_KEY — already in .config) ─────────
    # Cross-architecture consensus: different training data = different failure modes
    "openrouter_gpt4o": {
        "model_id": "openai/gpt-4o",
        "provider": "openrouter",
        "name": "GPT-4o via OpenRouter",
        "rpd": None,
        "rpm": None,
        "tpm": None,
        "supports_vision": True,
        "supports_thinking": False,
        "description": "OpenAI's top vision model via OpenRouter. Strongest cross-architecture consensus partner.",
        "suggested_sleep": 0
    },
    "openrouter_gemini": {
        "model_id": "google/gemini-2.0-flash-001",
        "provider": "openrouter",
        "name": "Gemini 2.0 Flash via OpenRouter",
        "rpd": None,
        "rpm": None,
        "tpm": None,
        "supports_vision": True,
        "supports_thinking": False,
        "description": "Google's fast vision model. Cheap 3rd-opinion for high-stakes DISPUTED events.",
        "suggested_sleep": 0
    }
}


def get_analyst_model(key: str = "llama_4_scout") -> dict:
    """
    Return the full model config dict for a given model key.

    Available keys:
      Groq (free):     'llama_4_scout' (vision), 'llama_3_3_70b', 'llama_3_1_8b'
      OpenRouter:      'openrouter_gpt4o', 'openrouter_gemini'
    """
    return MARITIME_ANALYST_MODELS.get(key, MARITIME_ANALYST_MODELS["llama_4_scout"])
