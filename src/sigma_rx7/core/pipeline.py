"""Main SigmaRx7 pipeline implementation."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog
import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker

from .config import Config
from ..ingestion.fhir_ingestion import FHIRIngestionEngine
from ..normalization.rxnorm import RxNormNormalizer
from ..rules.overlap_detector import OverlapDetector
from ..rules.generic_recommender import GenericRecommender
from ..rules.payer_rules import PayerRulesEngine
from ..ml.denial_predictor import DenialPredictor
from ..export.fhir_exporter import FHIRExporter
from ..export.feature_exporter import FeatureExporter

logger = structlog.get_logger()


class SigmaRx7Pipeline:
    """Main pipeline orchestrator for SigmaRx7 FHIR processing."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the pipeline with configuration."""
        self.config = config or Config()
        self.engine = None
        self.session_factory = None
        self._setup_database()
        self._initialize_components()
    
    def _setup_database(self):
        """Initialize database connection."""
        try:
            self.engine = create_engine(
                self.config.database.connection_string,
                echo=False,
                pool_pre_ping=True
            )
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.fhir_ingestion = FHIRIngestionEngine(self.config, self.engine)
        self.rxnorm_normalizer = RxNormNormalizer(self.config)
        self.overlap_detector = OverlapDetector(self.config)
        self.generic_recommender = GenericRecommender(self.config)
        self.payer_rules = PayerRulesEngine(self.config)
        self.denial_predictor = DenialPredictor(self.config) if self.config.pipeline.enable_ml else None
        self.fhir_exporter = FHIRExporter(self.config)
        self.feature_exporter = FeatureExporter(self.config)
        
        logger.info("Pipeline components initialized")
    
    async def run_full_pipeline(
        self, 
        input_data_path: str,
        output_dir: str = "output",
        data_format: str = "synthea"
    ) -> Dict[str, Any]:
        """Run the complete SigmaRx7 pipeline."""
        logger.info(f"Starting SigmaRx7 pipeline with input: {input_data_path}")
        
        results = {
            "status": "started",
            "input_path": input_data_path,
            "output_dir": output_dir,
            "stages": {}
        }
        
        try:
            # Stage 1: FHIR Ingestion
            logger.info("Stage 1: FHIR Data Ingestion")
            ingestion_result = await self.fhir_ingestion.ingest_data(
                input_data_path, data_format
            )
            results["stages"]["ingestion"] = ingestion_result
            
            # Stage 2: Medication Normalization
            logger.info("Stage 2: RxNorm Medication Normalization")
            normalization_result = await self.rxnorm_normalizer.normalize_medications()
            results["stages"]["normalization"] = normalization_result
            
            # Stage 3: Overlap Detection
            if self.config.pipeline.enable_overlap_detection:
                logger.info("Stage 3: Medication Overlap Detection")
                overlap_result = await self.overlap_detector.detect_overlaps()
                results["stages"]["overlap_detection"] = overlap_result
            
            # Stage 4: Generic Recommendations
            if self.config.pipeline.enable_generic_recommendations:
                logger.info("Stage 4: Generic Medication Recommendations")
                generic_result = await self.generic_recommender.recommend_generics()
                results["stages"]["generic_recommendations"] = generic_result
            
            # Stage 5: Payer Rules Alignment
            logger.info("Stage 5: Payer Rules Alignment")
            payer_result = await self.payer_rules.apply_rules()
            results["stages"]["payer_rules"] = payer_result
            
            # Stage 6: ML Denial Prediction
            if self.denial_predictor and self.config.pipeline.enable_ml:
                logger.info("Stage 6: ML-based Denial Prediction")
                ml_result = await self.denial_predictor.predict_denials()
                results["stages"]["ml_prediction"] = ml_result
            
            # Stage 7: Export FHIR Bundles
            logger.info("Stage 7: FHIR Bundle Export")
            export_result = await self.fhir_exporter.export_bundles(output_dir)
            results["stages"]["fhir_export"] = export_result
            
            # Stage 8: Export Feature Tables
            logger.info("Stage 8: Feature Table Export")
            feature_result = await self.feature_exporter.export_features(output_dir)
            results["stages"]["feature_export"] = feature_result
            
            results["status"] = "completed"
            logger.info("SigmaRx7 pipeline completed successfully")
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return results
    
    def load_synthea_data(self, synthea_path: str) -> Dict[str, Any]:
        """Load Synthea synthetic data."""
        logger.info(f"Loading Synthea data from: {synthea_path}")
        return asyncio.run(
            self.fhir_ingestion.ingest_data(synthea_path, "synthea")
        )
    
    def load_forgerx_data(self, forgerx_path: str) -> Dict[str, Any]:
        """Load ForgeRx synthetic data."""
        logger.info(f"Loading ForgeRx data from: {forgerx_path}")
        return asyncio.run(
            self.fhir_ingestion.ingest_data(forgerx_path, "forgerx")
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        with self.session_factory() as session:
            # Get table counts and basic statistics
            try:
                # This would be implemented based on actual schema
                status = {
                    "database_connected": self.engine is not None,
                    "tables": {},
                    "pipeline_config": self.config.pipeline.model_dump()
                }
                
                # Add table statistics when implemented
                # status["tables"]["patients"] = session.execute("SELECT COUNT(*) FROM patients").scalar()
                # status["tables"]["medications"] = session.execute("SELECT COUNT(*) FROM medications").scalar()
                
                return status
            except Exception as e:
                logger.error(f"Error getting pipeline status: {e}")
                return {"error": str(e)}
    
    def create_database_schema(self):
        """Create database schema for SigmaRx7."""
        logger.info("Creating database schema")
        # This would create all necessary tables
        # For now, we'll just create the metadata
        metadata = MetaData()
        # Define tables here
        metadata.create_all(self.engine)
        logger.info("Database schema created")
    
    def close(self):
        """Clean up resources."""
        if self.engine:
            self.engine.dispose()
        logger.info("Pipeline resources cleaned up")