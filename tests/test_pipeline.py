"""Test pipeline functionality."""

import pytest
from unittest.mock import Mock, patch
from sigma_rx7.core.pipeline import SigmaRx7Pipeline
from sigma_rx7.core.config import Config


class TestSigmaRx7Pipeline:
    """Test the main pipeline class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Config()
        config.database.host = "localhost"
        config.database.database = "test_db"
        return config
    
    @patch('sigma_rx7.core.pipeline.create_engine')
    @patch('sigma_rx7.core.pipeline.sessionmaker')
    def test_pipeline_initialization(self, mock_sessionmaker, mock_create_engine, mock_config):
        """Test pipeline initialization."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = Mock()
        mock_sessionmaker.return_value = mock_session_factory
        
        pipeline = SigmaRx7Pipeline(mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline.engine == mock_engine
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)
    
    @patch('sigma_rx7.core.pipeline.create_engine')
    @patch('sigma_rx7.core.pipeline.sessionmaker')
    def test_pipeline_components_initialization(self, mock_sessionmaker, mock_create_engine, mock_config):
        """Test that all pipeline components are initialized."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        pipeline = SigmaRx7Pipeline(mock_config)
        
        # Check that components are initialized
        assert pipeline.fhir_ingestion is not None
        assert pipeline.rxnorm_normalizer is not None
        assert pipeline.overlap_detector is not None
        assert pipeline.generic_recommender is not None
        assert pipeline.payer_rules is not None
        assert pipeline.fhir_exporter is not None
        assert pipeline.feature_exporter is not None
        
        # ML predictor should be initialized when enabled
        if mock_config.pipeline.enable_ml:
            assert pipeline.denial_predictor is not None
    
    @patch('sigma_rx7.core.pipeline.create_engine')
    @patch('sigma_rx7.core.pipeline.sessionmaker')
    def test_get_pipeline_status(self, mock_sessionmaker, mock_create_engine, mock_config):
        """Test getting pipeline status."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        pipeline = SigmaRx7Pipeline(mock_config)
        status = pipeline.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert "database_connected" in status
        assert "pipeline_config" in status
        assert status["database_connected"] is True
    
    @patch('sigma_rx7.core.pipeline.create_engine')
    @patch('sigma_rx7.core.pipeline.sessionmaker') 
    def test_pipeline_close(self, mock_sessionmaker, mock_create_engine, mock_config):
        """Test pipeline resource cleanup."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        pipeline = SigmaRx7Pipeline(mock_config)
        pipeline.close()
        
        mock_engine.dispose.assert_called_once()