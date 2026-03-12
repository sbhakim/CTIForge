"""Configuration management for CTIForge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


def load_yaml_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Environment-based settings, loaded from .env and system environment."""

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""

    cti_model: str = "gpt-4o"
    cti_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cti_similarity_threshold: float = 0.85
    cti_log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def available_providers(self) -> list[str]:
        """List which API providers have keys configured."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.deepseek_api_key:
            providers.append("deepseek")
        providers.append("local")  # always available via HuggingFace cache
        return providers


def get_settings() -> Settings:
    """Load settings from environment / .env file."""
    load_dotenv()
    return Settings()


def get_project_root() -> Path:
    """Return the project root directory."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
