"""Test configuration for SigmaRx7."""

import pytest
import tempfile
from pathlib import Path
from sigma_rx7.core.config import Config


class TestConfig:
    """Test configuration class."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = Config()
        
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.fhir.version == "R4"
        assert config.pipeline.enable_ml is True
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
        database:
          host: testhost
          port: 5433
        pipeline:
          enable_ml: false
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            assert config.database.host == "testhost"
            assert config.database.port == 5433
            assert config.pipeline.enable_ml is False
        finally:
            Path(config_path).unlink()
    
    def test_config_to_yaml(self):
        """Test saving configuration to YAML."""
        config = Config()
        config.database.host = "newhost"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            config.to_yaml(output_path)
            
            # Load back and verify
            loaded_config = Config.from_yaml(output_path)
            assert loaded_config.database.host == "newhost"
        finally:
            Path(output_path).unlink()
    
    def test_database_connection_string(self):
        """Test database connection string generation."""
        config = Config()
        config.database.username = "testuser"
        config.database.password = "testpass"
        config.database.host = "testhost"
        config.database.port = 5432
        config.database.database = "testdb"
        
        expected = "postgresql://testuser:testpass@testhost:5432/testdb"
        assert config.database.connection_string == expected