"""Medication overlap detection for SigmaRx7."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class MedicationOverlap:
    """Represents a medication overlap."""
    patient_id: str
    medication1_id: str
    medication1_name: str
    medication2_id: str
    medication2_name: str
    overlap_start: datetime
    overlap_end: datetime
    overlap_days: int
    overlap_type: str  # therapeutic_duplication, interaction, contraindication
    severity: str  # low, moderate, high, critical
    recommendation: str
    confidence_score: float


@dataclass
class TherapeuticClass:
    """Therapeutic classification."""
    code: str
    name: str
    level: int  # 1=broad, 2=intermediate, 3=specific


class OverlapDetector:
    """Medication overlap detection engine."""
    
    def __init__(self, config):
        """Initialize overlap detector."""
        self.config = config
        self._therapeutic_classes = self._load_therapeutic_classes()
        self._interaction_rules = self._load_interaction_rules()
    
    async def detect_overlaps(self) -> Dict[str, Any]:
        """Detect medication overlaps for all patients."""
        logger.info("Starting medication overlap detection")
        
        result = {
            "total_patients": 0,
            "patients_with_overlaps": 0,
            "total_overlaps": 0,
            "overlaps_by_severity": {"low": 0, "moderate": 0, "high": 0, "critical": 0},
            "overlaps_by_type": {},
            "overlaps": [],
            "recommendations": []
        }
        
        try:
            # Get patient medication data
            patient_medications = await self._get_patient_medications()
            result["total_patients"] = len(patient_medications)
            
            # Process each patient
            for patient_id, medications in patient_medications.items():
                patient_overlaps = await self._detect_patient_overlaps(patient_id, medications)
                
                if patient_overlaps:
                    result["patients_with_overlaps"] += 1
                    result["overlaps"].extend(patient_overlaps)
                    
                    # Update statistics
                    for overlap in patient_overlaps:
                        result["total_overlaps"] += 1
                        result["overlaps_by_severity"][overlap.severity] += 1
                        
                        if overlap.overlap_type not in result["overlaps_by_type"]:
                            result["overlaps_by_type"][overlap.overlap_type] = 0
                        result["overlaps_by_type"][overlap.overlap_type] += 1
                        
                        # Add recommendations
                        if overlap.severity in ["high", "critical"]:
                            result["recommendations"].append({
                                "patient_id": patient_id,
                                "type": "overlap_resolution",
                                "priority": overlap.severity,
                                "message": overlap.recommendation,
                                "medications": [overlap.medication1_name, overlap.medication2_name]
                            })
            
            # Store overlap results
            await self._store_overlap_results(result["overlaps"])
            
            logger.info(f"Overlap detection completed: {result['total_overlaps']} overlaps found in {result['patients_with_overlaps']} patients")
        
        except Exception as e:
            logger.error(f"Overlap detection failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _get_patient_medications(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get medication data for all patients."""
        # Placeholder - would query actual database
        # For demo purposes, return sample data
        return {
            "patient_001": [
                {
                    "id": "med_001",
                    "name": "Acetaminophen 500mg",
                    "rxcui": "161",
                    "start_date": datetime(2024, 1, 1),
                    "end_date": datetime(2024, 1, 31),
                    "therapeutic_class": "analgesic"
                },
                {
                    "id": "med_002", 
                    "name": "Ibuprofen 400mg",
                    "rxcui": "5640",
                    "start_date": datetime(2024, 1, 15),
                    "end_date": datetime(2024, 2, 15),
                    "therapeutic_class": "nsaid"
                }
            ],
            "patient_002": [
                {
                    "id": "med_003",
                    "name": "Lisinopril 10mg",
                    "rxcui": "29046",
                    "start_date": datetime(2024, 1, 1),
                    "end_date": datetime(2024, 12, 31),
                    "therapeutic_class": "ace_inhibitor"
                },
                {
                    "id": "med_004",
                    "name": "Enalapril 5mg", 
                    "rxcui": "3827",
                    "start_date": datetime(2024, 6, 1),
                    "end_date": datetime(2024, 12, 31),
                    "therapeutic_class": "ace_inhibitor"
                }
            ]
        }
    
    async def _detect_patient_overlaps(self, patient_id: str, medications: List[Dict[str, Any]]) -> List[MedicationOverlap]:
        """Detect overlaps for a single patient."""
        overlaps = []
        
        # Compare each pair of medications
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                med1 = medications[i]
                med2 = medications[j]
                
                # Check for temporal overlap
                temporal_overlap = self._calculate_temporal_overlap(med1, med2)
                if not temporal_overlap:
                    continue
                
                # Check for therapeutic overlap
                therapeutic_overlap = await self._check_therapeutic_overlap(med1, med2)
                if therapeutic_overlap:
                    overlap = MedicationOverlap(
                        patient_id=patient_id,
                        medication1_id=med1["id"],
                        medication1_name=med1["name"],
                        medication2_id=med2["id"],
                        medication2_name=med2["name"],
                        overlap_start=temporal_overlap["start"],
                        overlap_end=temporal_overlap["end"],
                        overlap_days=temporal_overlap["days"],
                        overlap_type=therapeutic_overlap["type"],
                        severity=therapeutic_overlap["severity"],
                        recommendation=therapeutic_overlap["recommendation"],
                        confidence_score=therapeutic_overlap["confidence"]
                    )
                    overlaps.append(overlap)
        
        return overlaps
    
    def _calculate_temporal_overlap(self, med1: Dict[str, Any], med2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate temporal overlap between two medications."""
        start1, end1 = med1["start_date"], med1["end_date"]
        start2, end2 = med2["start_date"], med2["end_date"]
        
        # Find overlap period
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days + 1
            return {
                "start": overlap_start,
                "end": overlap_end,
                "days": overlap_days
            }
        
        return None
    
    async def _check_therapeutic_overlap(self, med1: Dict[str, Any], med2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for therapeutic overlap between medications."""
        rxcui1 = med1.get("rxcui")
        rxcui2 = med2.get("rxcui")
        class1 = med1.get("therapeutic_class")
        class2 = med2.get("therapeutic_class")
        
        # Check for therapeutic duplication
        if class1 and class2 and class1 == class2:
            return {
                "type": "therapeutic_duplication",
                "severity": "moderate",
                "recommendation": f"Consider discontinuing one of the {class1} medications to avoid duplication",
                "confidence": 0.8
            }
        
        # Check interaction rules
        interaction = self._check_drug_interaction(rxcui1, rxcui2)
        if interaction:
            return interaction
        
        # Check contraindications
        contraindication = self._check_contraindication(med1, med2)
        if contraindication:
            return contraindication
        
        return None
    
    def _check_drug_interaction(self, rxcui1: str, rxcui2: str) -> Optional[Dict[str, Any]]:
        """Check for drug-drug interactions."""
        # Use interaction rules loaded at initialization
        interaction_key = tuple(sorted([rxcui1, rxcui2]))
        
        if interaction_key in self._interaction_rules:
            rule = self._interaction_rules[interaction_key]
            return {
                "type": "interaction",
                "severity": rule["severity"],
                "recommendation": rule["recommendation"],
                "confidence": rule["confidence"]
            }
        
        return None
    
    def _check_contraindication(self, med1: Dict[str, Any], med2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for contraindications."""
        # Example contraindication rules
        contraindications = {
            ("ace_inhibitor", "ace_inhibitor"): {
                "severity": "high",
                "recommendation": "Avoid using multiple ACE inhibitors simultaneously",
                "confidence": 0.9
            }
        }
        
        class1 = med1.get("therapeutic_class")
        class2 = med2.get("therapeutic_class")
        
        contraindication_key = tuple(sorted([class1, class2]))
        if contraindication_key in contraindications:
            rule = contraindications[contraindication_key]
            return {
                "type": "contraindication",
                **rule
            }
        
        return None
    
    def _load_therapeutic_classes(self) -> Dict[str, TherapeuticClass]:
        """Load therapeutic classification data."""
        # In production, this would load from a database or file
        return {
            "analgesic": TherapeuticClass("N02", "Analgesics", 2),
            "nsaid": TherapeuticClass("M01A", "Anti-inflammatory products, non-steroids", 3),
            "ace_inhibitor": TherapeuticClass("C09A", "ACE inhibitors", 3),
            "beta_blocker": TherapeuticClass("C07", "Beta blocking agents", 2),
            "statin": TherapeuticClass("C10AA", "HMG CoA reductase inhibitors", 3)
        }
    
    def _load_interaction_rules(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Load drug interaction rules."""
        # In production, this would load from a comprehensive drug interaction database
        return {
            ("161", "5640"): {  # Acetaminophen + Ibuprofen
                "severity": "low",
                "recommendation": "Monitor for increased risk of gastrointestinal effects",
                "confidence": 0.7
            },
            ("29046", "3827"): {  # Lisinopril + Enalapril
                "severity": "high", 
                "recommendation": "Avoid concurrent use of multiple ACE inhibitors",
                "confidence": 0.95
            }
        }
    
    async def _store_overlap_results(self, overlaps: List[MedicationOverlap]):
        """Store overlap detection results to database."""
        logger.info(f"Storing {len(overlaps)} overlap detection results")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([
            {
                "patient_id": o.patient_id,
                "medication1_id": o.medication1_id,
                "medication1_name": o.medication1_name,
                "medication2_id": o.medication2_id,
                "medication2_name": o.medication2_name,
                "overlap_start": o.overlap_start,
                "overlap_end": o.overlap_end,
                "overlap_days": o.overlap_days,
                "overlap_type": o.overlap_type,
                "severity": o.severity,
                "recommendation": o.recommendation,
                "confidence_score": o.confidence_score
            }
            for o in overlaps
        ])
        
        # In a real implementation, this would use SQLAlchemy to store to database
        logger.info(f"Overlap detection summary:\n{df.groupby(['overlap_type', 'severity']).size()}")
    
    def get_overlap_stats(self) -> Dict[str, Any]:
        """Get overlap detection statistics."""
        return {
            "therapeutic_classes_loaded": len(self._therapeutic_classes),
            "interaction_rules_loaded": len(self._interaction_rules)
        }