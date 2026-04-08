from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_local_config() -> None:
    for path in (REPO_ROOT / ".config", PROJECT_ROOT / ".config"):
        merge_config_file(path)
    
    # Auto-discovery of standalone .txt cookie files in config/
    if not os.getenv("YOUTUBE_INTAKE_YT_COOKIES"):
        config_dir = PROJECT_ROOT / "config"
        if config_dir.is_dir():
            for txt_file in config_dir.glob("*.txt"):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    if text.strip().startswith("# Netscape"):
                        os.environ["YOUTUBE_INTAKE_YT_COOKIES"] = text
                        break
                except Exception:
                    pass


def read_env(primary: str, legacy: str | None = None) -> str:
    value = os.getenv(primary, "").strip()
    if value:
        return value
    if legacy:
        return os.getenv(legacy, "").strip()
    return ""


def merge_config_file(path: Path) -> None:
    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    current_key: str | None = None
    current_value_lines: list[str] = []
    quote_char: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        
        if current_key is None:
            # Look for new key=value pair
            if not line or line.startswith("#") or "=" not in line:
                continue
            
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            if not value:
                os.environ.setdefault(key, "")
                continue
            
            # Check if value is quoted and potentially multi-line
            if value[0] in ('"', "'"):
                quote_char = value[0]
                if len(value) > 1 and value.endswith(quote_char):
                    # Single line quoted value
                    os.environ.setdefault(key, value[1:-1])
                else:
                    # Start of a multi-line quoted value
                    current_key = key
                    current_value_lines.append(value[1:])
            else:
                # Standard single-line value
                os.environ.setdefault(key, value)
        else:
            # We are inside a multi-line quoted value
            if quote_char and raw_line.endswith(quote_char):
                # End of multi-line value
                current_value_lines.append(raw_line[:-1])
                os.environ.setdefault(current_key, "\n".join(current_value_lines))
                current_key = None
                current_value_lines = []
                quote_char = None
            else:
                # Continue multi-line value
                current_value_lines.append(raw_line)
