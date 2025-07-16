"""Configuration settings for memloader."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"


class OllamaModel(str, Enum):
    """Supported Ollama models."""
    NUEXTRACT = "nuextract"
    LLAMA_3_2_1B = "llama3.2:1b"
    LLAMA_3_2_3B = "llama3.2:3b"
    MISTRAL_SMALL = "mistral-small"
    GEMMA_2B = "gemma:2b"


class OpenAIModel(str, Enum):
    """Supported OpenAI models."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"


class Settings(BaseSettings):
    """Application settings."""
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    ollama_model: OllamaModel = Field(default=OllamaModel.NUEXTRACT)
    openai_model: OpenAIModel = Field(default=OpenAIModel.GPT_4O_MINI)
    ollama_base_url: str = Field(default="http://localhost:11434")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    mem0_api_key: Optional[str] = Field(default=None)
    
    # Processing Configuration
    chunk_size: int = Field(default=1500)  # Tokens per chunk
    chunk_overlap: int = Field(default=200)  # Overlap between chunks
    batch_size: int = Field(default=100)  # Memories per batch to mem0
    confidence_threshold: float = Field(default=0.7)  # Minimum confidence for memories
    
    # Memory Categories
    memory_categories: list[str] = Field(default=[
        "preference",
        "fact",
        "pattern",
        "goal",
        "skill",
        "relationship",
        "context",
        "decision_criteria"
    ])
    
    # Rate Limiting
    requests_per_minute: int = Field(default=60)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    class Config:
        env_file = ".env"
        env_prefix = "MEMLOADER_"


# Global settings instance
settings = Settings()