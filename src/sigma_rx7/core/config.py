"""Configuration management for SigmaRx7 pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="sigma_rx7")
    username: str = Field(default="postgres")
    password: str = Field(default="")
    db_schema: str = Field(default="public", alias="schema")
    
    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class FHIRConfig(BaseModel):
    """FHIR processing configuration."""
    base_url: Optional[str] = None
    version: str = Field(default="R4")
    batch_size: int = Field(default=100)
    timeout: int = Field(default=30)


class RxNormConfig(BaseModel):
    """RxNorm API configuration."""
    api_url: str = Field(default="https://rxnav.nlm.nih.gov/REST")
    timeout: int = Field(default=10)
    rate_limit: float = Field(default=0.5)  # seconds between requests


class MLConfig(BaseModel):
    """Machine learning configuration."""
    model_path: str = Field(default="models/denial_predictor.joblib")
    feature_columns: list = Field(default_factory=list)
    target_column: str = Field(default="denial_probability")
    train_test_split: float = Field(default=0.2)


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    name: str = Field(default="sigma-rx7-pipeline")
    version: str = Field(default="1.0")
    parallel_workers: int = Field(default=4)
    chunk_size: int = Field(default=1000)
    enable_ml: bool = Field(default=True)
    enable_overlap_detection: bool = Field(default=True)
    enable_generic_recommendations: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class for SigmaRx7."""
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    fhir: FHIRConfig = Field(default_factory=FHIRConfig)
    rxnorm: RxNormConfig = Field(default_factory=RxNormConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_data = {}
        
        # Database config from environment
        if os.getenv("DATABASE_URL"):
            # Parse database URL if provided
            db_config = {}
            db_url = os.getenv("DATABASE_URL")
            # Simple parsing - in production would use sqlalchemy.engine.url.make_url
            if "://" in db_url:
                parts = db_url.split("://")[1].split("/")
                if "@" in parts[0]:
                    auth, host = parts[0].split("@")
                    if ":" in auth:
                        db_config["username"], db_config["password"] = auth.split(":")
                    if ":" in host:
                        db_config["host"], db_config["port"] = host.split(":")
                        db_config["port"] = int(db_config["port"])
                    else:
                        db_config["host"] = host
                if len(parts) > 1:
                    db_config["database"] = parts[1]
            config_data["database"] = db_config
        
        # Other env vars
        env_mappings = {
            "FHIR_BASE_URL": ("fhir", "base_url"),
            "RXNORM_API_URL": ("rxnorm", "api_url"),
            "ML_MODEL_PATH": ("ml", "model_path"),
            "PIPELINE_WORKERS": ("pipeline", "parallel_workers"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if section not in config_data:
                    config_data[section] = {}
                # Convert to appropriate type
                if key in ["parallel_workers", "port", "batch_size", "timeout"]:
                    value = int(value)
                elif key in ["rate_limit", "train_test_split"]:
                    value = float(value)
                elif key in ["enable_ml", "enable_overlap_detection", "enable_generic_recommendations"]:
                    value = value.lower() in ("true", "1", "yes")
                config_data[section][key] = value
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.model_dump()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {output_path}")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with fallback strategy."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    
    # Try default config locations
    default_paths = [
        "config/sigma_rx7.yaml",
        "sigma_rx7.yaml", 
        "/etc/sigma_rx7/config.yaml"
    ]
    
    for path in default_paths:
        if Path(path).exists():
            return Config.from_yaml(path)
    
    # Fall back to environment variables
    logger.info("No config file found, loading from environment variables")
    return Config.from_env()