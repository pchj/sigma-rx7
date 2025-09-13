"""FHIR Bundle export functionality for SigmaRx7."""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.patient import Patient
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.medication import Medication
from fhir.resources.observation import Observation
from fhir.resources.organization import Organization
from fhir.resources.practitioner import Practitioner

logger = structlog.get_logger()


class FHIRExporter:
    """FHIR Bundle export engine."""
    
    def __init__(self, config):
        """Initialize FHIR exporter."""
        self.config = config
        self.export_format = "json"  # Could be xml, json, or both
    
    async def export_bundles(self, output_dir: str = "output") -> Dict[str, Any]:
        """Export processed data as FHIR bundles."""
        logger.info(f"Starting FHIR bundle export to {output_dir}")
        
        result = {
            "output_directory": output_dir,
            "bundles_exported": 0,
            "patients_exported": 0,
            "total_resources": 0,
            "export_files": [],
            "bundle_types": {}
        }
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export different types of bundles
            bundle_types = [
                ("patient_bundles", self._export_patient_bundles),
                ("medication_bundles", self._export_medication_bundles),
                ("analysis_bundles", self._export_analysis_bundles)
            ]
            
            for bundle_type, export_func in bundle_types:
                bundle_result = await export_func(output_path)
                result["bundle_types"][bundle_type] = bundle_result
                result["bundles_exported"] += bundle_result.get("bundles_count", 0)
                result["total_resources"] += bundle_result.get("resources_count", 0)
                result["export_files"].extend(bundle_result.get("files", []))
            
            # Export summary bundle
            summary_result = await self._export_summary_bundle(output_path, result)
            result["export_files"].append(summary_result["file"])
            
            logger.info(f"FHIR export completed: {result['bundles_exported']} bundles with {result['total_resources']} resources")
        
        except Exception as e:
            logger.error(f"FHIR export failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _export_patient_bundles(self, output_path: Path) -> Dict[str, Any]:
        """Export patient-specific FHIR bundles."""
        logger.info("Exporting patient bundles")
        
        result = {
            "bundles_count": 0,
            "resources_count": 0,
            "files": []
        }
        
        # Get patient data
        patients_data = await self._get_patients_data()
        
        for patient_id, patient_info in patients_data.items():
            # Create patient bundle
            bundle = await self._create_patient_bundle(patient_id, patient_info)
            
            # Save bundle to file
            bundle_file = output_path / f"patient_bundle_{patient_id}.json"
            with open(bundle_file, 'w') as f:
                json.dump(bundle.model_dump(), f, indent=2, default=str)
            
            result["bundles_count"] += 1
            result["resources_count"] += len(bundle.entry) if bundle.entry else 0
            result["files"].append(str(bundle_file))
        
        return result
    
    async def _export_medication_bundles(self, output_path: Path) -> Dict[str, Any]:
        """Export medication-focused FHIR bundles."""
        logger.info("Exporting medication bundles")
        
        result = {
            "bundles_count": 0,
            "resources_count": 0,
            "files": []
        }
        
        # Get medication data with recommendations
        medication_data = await self._get_medication_analysis_data()
        
        # Create medication analysis bundle
        bundle = await self._create_medication_analysis_bundle(medication_data)
        
        # Save bundle to file
        bundle_file = output_path / "medication_analysis_bundle.json"
        with open(bundle_file, 'w') as f:
            json.dump(bundle.model_dump(), f, indent=2, default=str)
        
        result["bundles_count"] = 1
        result["resources_count"] = len(bundle.entry) if bundle.entry else 0
        result["files"].append(str(bundle_file))
        
        return result
    
    async def _export_analysis_bundles(self, output_path: Path) -> Dict[str, Any]:
        """Export analysis results as FHIR bundles."""
        logger.info("Exporting analysis bundles")
        
        result = {
            "bundles_count": 0,
            "resources_count": 0,
            "files": []
        }
        
        # Export different analysis types
        analysis_types = [
            ("overlap_analysis", self._create_overlap_analysis_bundle),
            ("generic_recommendations", self._create_generic_recommendations_bundle),
            ("payer_rules_analysis", self._create_payer_rules_bundle),
            ("denial_predictions", self._create_denial_predictions_bundle)
        ]
        
        for analysis_name, create_func in analysis_types:
            try:
                bundle = await create_func()
                
                # Save bundle to file
                bundle_file = output_path / f"{analysis_name}_bundle.json"
                with open(bundle_file, 'w') as f:
                    json.dump(bundle.model_dump(), f, indent=2, default=str)
                
                result["bundles_count"] += 1
                result["resources_count"] += len(bundle.entry) if bundle.entry else 0
                result["files"].append(str(bundle_file))
            
            except Exception as e:
                logger.error(f"Failed to export {analysis_name}: {e}")
        
        return result
    
    async def _create_patient_bundle(self, patient_id: str, patient_info: Dict[str, Any]) -> Bundle:
        """Create a FHIR bundle for a patient."""
        bundle_id = f"patient-bundle-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        bundle = Bundle(
            id=bundle_id,
            type="collection",
            timestamp=datetime.now().isoformat(),
            meta={
                "versionId": "1",
                "lastUpdated": datetime.now().isoformat(),
                "profile": ["http://sigma-rx7.com/fhir/StructureDefinition/PatientBundle"]
            }
        )
        
        entries = []
        
        # Add patient resource
        patient_resource = self._create_patient_resource(patient_id, patient_info)
        entries.append(BundleEntry(
            resource=patient_resource,
            fullUrl=f"Patient/{patient_id}"
        ))
        
        # Add medication requests
        for medication in patient_info.get("medications", []):
            med_request = self._create_medication_request_resource(patient_id, medication)
            entries.append(BundleEntry(
                resource=med_request,
                fullUrl=f"MedicationRequest/{medication['id']}"
            ))
        
        # Add observations (e.g., analysis results)
        observations = await self._create_patient_observations(patient_id, patient_info)
        for obs in observations:
            entries.append(BundleEntry(
                resource=obs,
                fullUrl=f"Observation/{obs.id}"
            ))
        
        bundle.entry = entries
        return bundle
    
    async def _create_medication_analysis_bundle(self, medication_data: Dict[str, Any]) -> Bundle:
        """Create a bundle with medication analysis results."""
        bundle = Bundle(
            id=f"medication-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        entries = []
        
        # Add medication resources with analysis
        for med_id, med_info in medication_data.items():
            # Create medication resource
            medication = Medication(
                id=med_id,
                code={
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": med_info.get("rxcui", ""),
                        "display": med_info.get("name", "")
                    }]
                }
            )
            
            entries.append(BundleEntry(
                resource=medication,
                fullUrl=f"Medication/{med_id}"
            ))
        
        bundle.entry = entries
        return bundle
    
    async def _create_overlap_analysis_bundle(self) -> Bundle:
        """Create bundle with overlap analysis results."""
        bundle = Bundle(
            id=f"overlap-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        # Get overlap data
        overlap_data = await self._get_overlap_analysis_data()
        
        entries = []
        for overlap in overlap_data:
            # Create observation for each overlap
            observation = Observation(
                id=f"overlap-{overlap['id']}",
                status="final",
                category=[{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey"
                    }]
                }],
                code={
                    "coding": [{
                        "system": "http://sigma-rx7.com/fhir/CodeSystem/analysis-type",
                        "code": "medication-overlap",
                        "display": "Medication Overlap Analysis"
                    }]
                },
                subject={
                    "reference": f"Patient/{overlap['patient_id']}"
                },
                valueString=overlap["description"]
            )
            
            entries.append(BundleEntry(
                resource=observation,
                fullUrl=f"Observation/overlap-{overlap['id']}"
            ))
        
        bundle.entry = entries
        return bundle
    
    async def _create_generic_recommendations_bundle(self) -> Bundle:
        """Create bundle with generic medication recommendations."""
        bundle = Bundle(
            id=f"generic-recommendations-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        # Get recommendation data
        recommendations_data = await self._get_generic_recommendations_data()
        
        entries = []
        for rec in recommendations_data:
            # Create observation for each recommendation
            observation = Observation(
                id=f"generic-rec-{rec['id']}",
                status="final",
                category=[{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "survey",
                        "display": "Survey"
                    }]
                }],
                code={
                    "coding": [{
                        "system": "http://sigma-rx7.com/fhir/CodeSystem/analysis-type",
                        "code": "generic-recommendation",
                        "display": "Generic Medication Recommendation"
                    }]
                },
                subject={
                    "reference": f"Patient/{rec['patient_id']}"
                },
                component=[
                    {
                        "code": {
                            "coding": [{
                                "system": "http://sigma-rx7.com/fhir/CodeSystem/recommendation-metric",
                                "code": "cost-savings",
                                "display": "Cost Savings"
                            }]
                        },
                        "valueQuantity": {
                            "value": rec["cost_savings"],
                            "unit": "USD",
                            "system": "http://unitsofmeasure.org",
                            "code": "USD"
                        }
                    }
                ]
            )
            
            entries.append(BundleEntry(
                resource=observation,
                fullUrl=f"Observation/generic-rec-{rec['id']}"
            ))
        
        bundle.entry = entries
        return bundle
    
    async def _create_payer_rules_bundle(self) -> Bundle:
        """Create bundle with payer rules analysis."""
        bundle = Bundle(
            id=f"payer-rules-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        # Implementation would create observations for rule violations
        entries = []
        bundle.entry = entries
        return bundle
    
    async def _create_denial_predictions_bundle(self) -> Bundle:
        """Create bundle with denial predictions."""
        bundle = Bundle(
            id=f"denial-predictions-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        # Implementation would create observations for predictions
        entries = []
        bundle.entry = entries
        return bundle
    
    async def _export_summary_bundle(self, output_path: Path, export_result: Dict[str, Any]) -> Dict[str, str]:
        """Export a summary bundle with export statistics."""
        summary_bundle = Bundle(
            id=f"export-summary-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            type="collection",
            timestamp=datetime.now().isoformat()
        )
        
        # Create summary observation
        summary_obs = Observation(
            id="export-summary",
            status="final",
            code={
                "coding": [{
                    "system": "http://sigma-rx7.com/fhir/CodeSystem/analysis-type",
                    "code": "export-summary",
                    "display": "Export Summary"
                }]
            },
            component=[
                {
                    "code": {"text": "Total Bundles Exported"},
                    "valueInteger": export_result["bundles_exported"]
                },
                {
                    "code": {"text": "Total Resources"},
                    "valueInteger": export_result["total_resources"]
                }
            ]
        )
        
        summary_bundle.entry = [BundleEntry(
            resource=summary_obs,
            fullUrl="Observation/export-summary"
        )]
        
        # Save summary bundle
        summary_file = output_path / "export_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_bundle.model_dump(), f, indent=2, default=str)
        
        return {"file": str(summary_file)}
    
    def _create_patient_resource(self, patient_id: str, patient_info: Dict[str, Any]) -> Patient:
        """Create a FHIR Patient resource."""
        demographics = patient_info.get("demographics", {})
        
        patient = Patient(
            id=patient_id,
            active=True,
            name=[{
                "use": "official",
                "family": f"Patient{patient_id}",
                "given": ["Test"]
            }],
            gender=demographics.get("gender", "unknown").lower(),
            birthDate=self._calculate_birth_date(demographics.get("age", 30))
        )
        
        return patient
    
    def _create_medication_request_resource(self, patient_id: str, medication: Dict[str, Any]) -> MedicationRequest:
        """Create a FHIR MedicationRequest resource."""
        med_request = MedicationRequest(
            id=medication["id"],
            status="active",
            intent="order",
            medicationCodeableConcept={
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.get("rxcui", ""),
                    "display": medication.get("name", "")
                }]
            },
            subject={
                "reference": f"Patient/{patient_id}"
            },
            dispenseRequest={
                "quantity": {
                    "value": medication.get("quantity", 30),
                    "unit": "tablets"
                },
                "expectedSupplyDuration": {
                    "value": medication.get("days_supply", 30),
                    "unit": "days",
                    "system": "http://unitsofmeasure.org",
                    "code": "d"
                }
            }
        )
        
        return med_request
    
    async def _create_patient_observations(self, patient_id: str, patient_info: Dict[str, Any]) -> List[Observation]:
        """Create observation resources for patient analysis results."""
        observations = []
        
        # Example: Create observation for analysis results
        # This would be expanded based on actual analysis results
        
        return observations
    
    def _calculate_birth_date(self, age: int) -> str:
        """Calculate birth date from age."""
        birth_year = datetime.now().year - age
        return f"{birth_year}-01-01"
    
    async def _get_patients_data(self) -> Dict[str, Dict[str, Any]]:
        """Get patient data for export."""
        # Placeholder - would query actual database
        return {
            "patient_001": {
                "demographics": {"age": 45, "gender": "M"},
                "medications": [
                    {"id": "med_001", "name": "Ozempic 0.5mg", "rxcui": "1991302", "quantity": 1, "days_supply": 28}
                ]
            }
        }
    
    async def _get_medication_analysis_data(self) -> Dict[str, Dict[str, Any]]:
        """Get medication analysis data for export."""
        return {
            "med_001": {"name": "Ozempic 0.5mg", "rxcui": "1991302"}
        }
    
    async def _get_overlap_analysis_data(self) -> List[Dict[str, Any]]:
        """Get overlap analysis data for export."""
        return [
            {"id": "overlap_001", "patient_id": "patient_001", "description": "No overlaps detected"}
        ]
    
    async def _get_generic_recommendations_data(self) -> List[Dict[str, Any]]:
        """Get generic recommendations data for export."""
        return [
            {"id": "rec_001", "patient_id": "patient_001", "cost_savings": 120.50}
        ]