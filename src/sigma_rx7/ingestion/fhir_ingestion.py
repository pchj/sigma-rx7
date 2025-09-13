"""FHIR data ingestion engine for SigmaRx7."""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import structlog
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.medication import Medication
from fhir.resources.observation import Observation
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

logger = structlog.get_logger()


class FHIRIngestionEngine:
    """Engine for ingesting FHIR data from various sources."""
    
    def __init__(self, config, engine: Engine):
        """Initialize the FHIR ingestion engine."""
        self.config = config
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine)
    
    async def ingest_data(
        self, 
        data_path: str, 
        data_format: str = "synthea"
    ) -> Dict[str, Any]:
        """Ingest FHIR data from specified path and format."""
        logger.info(f"Ingesting {data_format} data from: {data_path}")
        
        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        if data_format.lower() == "synthea":
            return await self._ingest_synthea_data(data_path_obj)
        elif data_format.lower() == "forgerx":
            return await self._ingest_forgerx_data(data_path_obj)
        elif data_format.lower() == "fhir_bundle":
            return await self._ingest_fhir_bundle(data_path_obj)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    async def _ingest_synthea_data(self, data_path: Path) -> Dict[str, Any]:
        """Ingest Synthea synthetic data."""
        logger.info("Processing Synthea data format")
        
        result = {
            "format": "synthea",
            "patients_processed": 0,
            "medications_processed": 0,
            "observations_processed": 0,
            "files_processed": []
        }
        
        # Synthea typically outputs FHIR bundles as NDJSON files
        if data_path.is_file() and data_path.suffix == ".json":
            # Single bundle file
            bundle_result = await self._process_fhir_bundle_file(data_path)
            result.update(bundle_result)
            result["files_processed"] = [str(data_path)]
        elif data_path.is_dir():
            # Directory of FHIR files
            fhir_files = list(data_path.glob("*.json"))
            for fhir_file in fhir_files:
                try:
                    bundle_result = await self._process_fhir_bundle_file(fhir_file)
                    result["patients_processed"] += bundle_result.get("patients_processed", 0)
                    result["medications_processed"] += bundle_result.get("medications_processed", 0)
                    result["observations_processed"] += bundle_result.get("observations_processed", 0)
                    result["files_processed"].append(str(fhir_file))
                except Exception as e:
                    logger.error(f"Error processing file {fhir_file}: {e}")
        
        logger.info(f"Synthea ingestion completed: {result}")
        return result
    
    async def _ingest_forgerx_data(self, data_path: Path) -> Dict[str, Any]:
        """Ingest ForgeRx synthetic data."""
        logger.info("Processing ForgeRx data format")
        
        # ForgeRx format handling (assuming similar to Synthea but with different structure)
        result = {
            "format": "forgerx",
            "patients_processed": 0,
            "medications_processed": 0,
            "files_processed": []
        }
        
        # Process ForgeRx files
        if data_path.is_file():
            forgerx_result = await self._process_forgerx_file(data_path)
            result.update(forgerx_result)
            result["files_processed"] = [str(data_path)]
        elif data_path.is_dir():
            forgerx_files = list(data_path.glob("*.json"))
            for forgerx_file in forgerx_files:
                try:
                    forgerx_result = await self._process_forgerx_file(forgerx_file)
                    result["patients_processed"] += forgerx_result.get("patients_processed", 0)
                    result["medications_processed"] += forgerx_result.get("medications_processed", 0)
                    result["files_processed"].append(str(forgerx_file))
                except Exception as e:
                    logger.error(f"Error processing ForgeRx file {forgerx_file}: {e}")
        
        return result
    
    async def _ingest_fhir_bundle(self, data_path: Path) -> Dict[str, Any]:
        """Ingest standard FHIR bundle."""
        logger.info("Processing standard FHIR bundle")
        return await self._process_fhir_bundle_file(data_path)
    
    async def _process_fhir_bundle_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single FHIR bundle file."""
        logger.debug(f"Processing FHIR bundle: {file_path}")
        
        result = {
            "patients_processed": 0,
            "medications_processed": 0,
            "observations_processed": 0,
            "errors": []
        }
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single bundles and NDJSON format
            if isinstance(data, dict) and data.get("resourceType") == "Bundle":
                bundle_result = await self._process_single_bundle(data)
                result.update(bundle_result)
            elif isinstance(data, list):
                # Array of resources
                for resource in data:
                    if resource.get("resourceType") == "Bundle":
                        bundle_result = await self._process_single_bundle(resource)
                        result["patients_processed"] += bundle_result.get("patients_processed", 0)
                        result["medications_processed"] += bundle_result.get("medications_processed", 0)
                        result["observations_processed"] += bundle_result.get("observations_processed", 0)
            else:
                # Try to parse as NDJSON
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            resource = json.loads(line)
                            if resource.get("resourceType") == "Bundle":
                                bundle_result = await self._process_single_bundle(resource)
                                result["patients_processed"] += bundle_result.get("patients_processed", 0)
                                result["medications_processed"] += bundle_result.get("medications_processed", 0)
                                result["observations_processed"] += bundle_result.get("observations_processed", 0)
        
        except Exception as e:
            logger.error(f"Error processing FHIR bundle {file_path}: {e}")
            result["errors"].append(str(e))
        
        return result
    
    async def _process_single_bundle(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single FHIR bundle."""
        result = {
            "patients_processed": 0,
            "medications_processed": 0,
            "observations_processed": 0
        }
        
        try:
            bundle = Bundle(**bundle_data)
            
            # Process each entry in the bundle
            if bundle.entry:
                for entry in bundle.entry:
                    if entry.resource:
                        resource_type = entry.resource.resource_type
                        
                        if resource_type == "Patient":
                            await self._store_patient(entry.resource)
                            result["patients_processed"] += 1
                        elif resource_type in ["MedicationRequest", "MedicationStatement"]:
                            await self._store_medication_request(entry.resource)
                            result["medications_processed"] += 1
                        elif resource_type == "Medication":
                            await self._store_medication(entry.resource)
                        elif resource_type == "Observation":
                            await self._store_observation(entry.resource)
                            result["observations_processed"] += 1
        
        except Exception as e:
            logger.error(f"Error processing bundle: {e}")
            raise
        
        return result
    
    async def _process_forgerx_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a ForgeRx format file."""
        # Placeholder implementation for ForgeRx format
        # This would be customized based on actual ForgeRx data structure
        logger.info(f"Processing ForgeRx file: {file_path}")
        
        result = {
            "patients_processed": 0,
            "medications_processed": 0
        }
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # ForgeRx specific processing logic would go here
            # For now, assume it's similar to FHIR bundle structure
            if isinstance(data, dict) and "patients" in data:
                result["patients_processed"] = len(data.get("patients", []))
            if isinstance(data, dict) and "medications" in data:
                result["medications_processed"] = len(data.get("medications", []))
        
        except Exception as e:
            logger.error(f"Error processing ForgeRx file {file_path}: {e}")
        
        return result
    
    async def _store_patient(self, patient_resource):
        """Store patient resource in database."""
        # Placeholder for patient storage logic
        logger.debug(f"Storing patient: {patient_resource.id}")
        # Implementation would use SQLAlchemy models to store in database
    
    async def _store_medication_request(self, medication_request):
        """Store medication request in database."""
        logger.debug(f"Storing medication request: {medication_request.id}")
        # Implementation would use SQLAlchemy models to store in database
    
    async def _store_medication(self, medication):
        """Store medication resource in database."""
        logger.debug(f"Storing medication: {medication.id}")
        # Implementation would use SQLAlchemy models to store in database
    
    async def _store_observation(self, observation):
        """Store observation resource in database."""
        logger.debug(f"Storing observation: {observation.id}")
        # Implementation would use SQLAlchemy models to store in database
    
    async def validate_fhir_resource(self, resource_data: Dict[str, Any]) -> bool:
        """Validate FHIR resource structure."""
        try:
            resource_type = resource_data.get("resourceType")
            if not resource_type:
                return False
            
            # Basic FHIR validation - in production would use proper FHIR validation
            required_fields = ["resourceType", "id"]
            return all(field in resource_data for field in required_fields)
        
        except Exception as e:
            logger.error(f"FHIR validation error: {e}")
            return False