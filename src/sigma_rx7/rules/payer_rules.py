"""Payer rules engine for SigmaRx7."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger()


class RuleType(Enum):
    """Types of payer rules."""
    PRIOR_AUTHORIZATION = "prior_authorization"
    STEP_THERAPY = "step_therapy"
    QUANTITY_LIMIT = "quantity_limit"
    AGE_LIMIT = "age_limit"
    DIAGNOSIS_REQUIREMENT = "diagnosis_requirement"
    FORMULARY_RESTRICTION = "formulary_restriction"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PayerRule:
    """Payer rule definition."""
    id: str
    name: str
    type: RuleType
    description: str
    criteria: Dict[str, Any]
    action: str
    severity: RuleSeverity
    payer_id: str
    effective_date: datetime
    expiration_date: Optional[datetime] = None


@dataclass
class RuleViolation:
    """Payer rule violation."""
    patient_id: str
    medication_id: str
    medication_name: str
    rule_id: str
    rule_name: str
    rule_type: RuleType
    severity: RuleSeverity
    violation_description: str
    recommendation: str
    estimated_denial_probability: float
    alternative_medications: List[str]


class PayerRulesEngine:
    """Payer rules alignment engine."""
    
    def __init__(self, config):
        """Initialize payer rules engine."""
        self.config = config
        self._payer_rules = self._load_payer_rules()
        self._formularies = self._load_formulary_data()
        self._step_therapy_protocols = self._load_step_therapy_protocols()
    
    async def apply_rules(self) -> Dict[str, Any]:
        """Apply payer rules to all patient medications."""
        logger.info("Starting payer rules alignment")
        
        result = {
            "total_patients": 0,
            "total_medications": 0,
            "total_violations": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
            "estimated_denials": 0,
            "violations": [],
            "recommendations": [],
            "coverage_analysis": {}
        }
        
        try:
            # Get patient medication data with payer information
            patient_data = await self._get_patient_medication_data()
            result["total_patients"] = len(patient_data)
            
            # Process each patient
            for patient_id, patient_info in patient_data.items():
                patient_violations = await self._check_patient_rules(patient_id, patient_info)
                
                result["violations"].extend(patient_violations)
                result["total_medications"] += len(patient_info["medications"])
                
                # Update statistics
                for violation in patient_violations:
                    result["total_violations"] += 1
                    
                    # Count by type
                    rule_type = violation.rule_type.value
                    if rule_type not in result["violations_by_type"]:
                        result["violations_by_type"][rule_type] = 0
                    result["violations_by_type"][rule_type] += 1
                    
                    # Count by severity
                    severity = violation.severity.value
                    if severity not in result["violations_by_severity"]:
                        result["violations_by_severity"][severity] = 0
                    result["violations_by_severity"][severity] += 1
                    
                    # Estimate denials
                    if violation.estimated_denial_probability > 0.7:
                        result["estimated_denials"] += 1
                    
                    # Generate recommendations
                    if violation.severity in [RuleSeverity.ERROR, RuleSeverity.CRITICAL]:
                        result["recommendations"].append({
                            "patient_id": patient_id,
                            "type": "rule_violation",
                            "priority": severity,
                            "message": violation.recommendation,
                            "medication": violation.medication_name,
                            "alternatives": violation.alternative_medications
                        })
            
            # Perform coverage analysis
            result["coverage_analysis"] = await self._analyze_coverage(patient_data)
            
            # Store rule violation results
            await self._store_rule_violations(result["violations"])
            
            logger.info(f"Payer rules analysis completed: {result['total_violations']} violations found, {result['estimated_denials']} estimated denials")
        
        except Exception as e:
            logger.error(f"Payer rules analysis failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _get_patient_medication_data(self) -> Dict[str, Dict[str, Any]]:
        """Get patient medication data with payer information."""
        # Placeholder - would query actual database
        # For demo purposes, return sample data
        return {
            "patient_001": {
                "demographics": {
                    "age": 45,
                    "gender": "M",
                    "diagnoses": ["E11.9", "I10"]  # Type 2 diabetes, hypertension
                },
                "insurance": {
                    "payer_id": "UHC_001",
                    "plan_type": "commercial",
                    "effective_date": datetime(2024, 1, 1)
                },
                "medications": [
                    {
                        "id": "med_001",
                        "name": "Ozempic 0.5mg",
                        "rxcui": "1991302",
                        "quantity": 1,
                        "days_supply": 28,
                        "diagnosis_codes": ["E11.9"],
                        "prescriber_id": "doc_001"
                    },
                    {
                        "id": "med_002",
                        "name": "Lipitor 40mg",
                        "rxcui": "617311",
                        "quantity": 30,
                        "days_supply": 30,
                        "diagnosis_codes": ["E78.5"],
                        "prescriber_id": "doc_001"
                    }
                ]
            },
            "patient_002": {
                "demographics": {
                    "age": 72,
                    "gender": "F",
                    "diagnoses": ["I10", "M79.3"]
                },
                "insurance": {
                    "payer_id": "MEDICARE_001",
                    "plan_type": "medicare",
                    "effective_date": datetime(2024, 1, 1)
                },
                "medications": [
                    {
                        "id": "med_003",
                        "name": "OxyContin 20mg",
                        "rxcui": "1049621",
                        "quantity": 60,
                        "days_supply": 30,
                        "diagnosis_codes": ["M79.3"],
                        "prescriber_id": "doc_002"
                    }
                ]
            }
        }
    
    async def _check_patient_rules(self, patient_id: str, patient_info: Dict[str, Any]) -> List[RuleViolation]:
        """Check payer rules for a single patient."""
        violations = []
        payer_id = patient_info["insurance"]["payer_id"]
        
        # Get applicable rules for this payer
        applicable_rules = [rule for rule in self._payer_rules if rule.payer_id == payer_id]
        
        for medication in patient_info["medications"]:
            for rule in applicable_rules:
                violation = await self._check_medication_rule(patient_id, patient_info, medication, rule)
                if violation:
                    violations.append(violation)
        
        return violations
    
    async def _check_medication_rule(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check a specific rule against a medication."""
        if rule.type == RuleType.PRIOR_AUTHORIZATION:
            return await self._check_prior_authorization(patient_id, patient_info, medication, rule)
        elif rule.type == RuleType.STEP_THERAPY:
            return await self._check_step_therapy(patient_id, patient_info, medication, rule)
        elif rule.type == RuleType.QUANTITY_LIMIT:
            return await self._check_quantity_limit(patient_id, patient_info, medication, rule)
        elif rule.type == RuleType.AGE_LIMIT:
            return await self._check_age_limit(patient_id, patient_info, medication, rule)
        elif rule.type == RuleType.DIAGNOSIS_REQUIREMENT:
            return await self._check_diagnosis_requirement(patient_id, patient_info, medication, rule)
        elif rule.type == RuleType.FORMULARY_RESTRICTION:
            return await self._check_formulary_restriction(patient_id, patient_info, medication, rule)
        
        return None
    
    async def _check_prior_authorization(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check prior authorization requirements."""
        rxcui = medication["rxcui"]
        
        # Check if medication requires PA
        if rxcui in rule.criteria.get("required_rxcuis", []):
            # Check if PA criteria are met
            age = patient_info["demographics"]["age"]
            diagnoses = patient_info["demographics"]["diagnoses"]
            
            pa_criteria = rule.criteria.get("pa_criteria", {})
            
            # Check age requirement
            if "min_age" in pa_criteria and age < pa_criteria["min_age"]:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Prior authorization required: Patient age {age} is below minimum age {pa_criteria['min_age']}",
                    recommendation="Consider alternative medication or obtain prior authorization",
                    estimated_denial_probability=0.8,
                    alternative_medications=rule.criteria.get("alternatives", [])
                )
            
            # Check diagnosis requirement
            required_diagnoses = pa_criteria.get("required_diagnoses", [])
            if required_diagnoses and not any(dx in diagnoses for dx in required_diagnoses):
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Prior authorization required: Missing required diagnosis. Required: {required_diagnoses}, Present: {diagnoses}",
                    recommendation="Add appropriate diagnosis code or obtain prior authorization",
                    estimated_denial_probability=0.9,
                    alternative_medications=rule.criteria.get("alternatives", [])
                )
        
        return None
    
    async def _check_step_therapy(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check step therapy requirements."""
        rxcui = medication["rxcui"]
        
        if rxcui in rule.criteria.get("step_therapy_drugs", []):
            required_first_line = rule.criteria.get("required_first_line", [])
            
            # Check patient medication history for first-line drugs
            # In production, this would query historical medication data
            patient_med_history = []  # Placeholder
            
            has_tried_first_line = any(med in patient_med_history for med in required_first_line)
            
            if not has_tried_first_line:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Step therapy violation: Must try first-line drugs before {medication['name']}",
                    recommendation=f"Try first-line medications: {', '.join(required_first_line)}",
                    estimated_denial_probability=0.95,
                    alternative_medications=required_first_line
                )
        
        return None
    
    async def _check_quantity_limit(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check quantity limit restrictions."""
        rxcui = medication["rxcui"]
        quantity = medication.get("quantity", 0)
        days_supply = medication.get("days_supply", 30)
        
        if rxcui in rule.criteria.get("quantity_limits", {}):
            limits = rule.criteria["quantity_limits"][rxcui]
            
            # Check quantity per fill
            max_quantity = limits.get("max_quantity_per_fill")
            if max_quantity and quantity > max_quantity:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Quantity limit exceeded: {quantity} exceeds maximum {max_quantity}",
                    recommendation=f"Reduce quantity to {max_quantity} or less",
                    estimated_denial_probability=0.7,
                    alternative_medications=[]
                )
            
            # Check days supply
            max_days_supply = limits.get("max_days_supply")
            if max_days_supply and days_supply > max_days_supply:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Days supply limit exceeded: {days_supply} exceeds maximum {max_days_supply}",
                    recommendation=f"Reduce days supply to {max_days_supply} or less",
                    estimated_denial_probability=0.6,
                    alternative_medications=[]
                )
        
        return None
    
    async def _check_age_limit(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check age-based restrictions."""
        rxcui = medication["rxcui"]
        age = patient_info["demographics"]["age"]
        
        if rxcui in rule.criteria.get("age_restrictions", {}):
            age_limits = rule.criteria["age_restrictions"][rxcui]
            
            min_age = age_limits.get("min_age")
            max_age = age_limits.get("max_age")
            
            if min_age and age < min_age:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Age restriction: Patient age {age} is below minimum {min_age}",
                    recommendation=f"Consider age-appropriate alternatives",
                    estimated_denial_probability=0.85,
                    alternative_medications=age_limits.get("alternatives", [])
                )
            
            if max_age and age > max_age:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Age restriction: Patient age {age} exceeds maximum {max_age}",
                    recommendation=f"Consider age-appropriate alternatives",
                    estimated_denial_probability=0.85,
                    alternative_medications=age_limits.get("alternatives", [])
                )
        
        return None
    
    async def _check_diagnosis_requirement(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check diagnosis requirements."""
        rxcui = medication["rxcui"]
        patient_diagnoses = patient_info["demographics"]["diagnoses"]
        medication_diagnoses = medication.get("diagnosis_codes", [])
        
        if rxcui in rule.criteria.get("diagnosis_requirements", {}):
            required_diagnoses = rule.criteria["diagnosis_requirements"][rxcui]
            
            # Check if any required diagnosis is present
            has_required_diagnosis = any(dx in patient_diagnoses + medication_diagnoses for dx in required_diagnoses)
            
            if not has_required_diagnosis:
                return RuleViolation(
                    patient_id=patient_id,
                    medication_id=medication["id"],
                    medication_name=medication["name"],
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    severity=rule.severity,
                    violation_description=f"Missing required diagnosis. Required: {required_diagnoses}",
                    recommendation="Add appropriate diagnosis code for indication",
                    estimated_denial_probability=0.9,
                    alternative_medications=[]
                )
        
        return None
    
    async def _check_formulary_restriction(self, patient_id: str, patient_info: Dict[str, Any], medication: Dict[str, Any], rule: PayerRule) -> Optional[RuleViolation]:
        """Check formulary restrictions."""
        rxcui = medication["rxcui"]
        payer_id = patient_info["insurance"]["payer_id"]
        
        formulary = self._formularies.get(payer_id, {})
        med_status = formulary.get(rxcui, {})
        
        if med_status.get("status") == "not_covered":
            return RuleViolation(
                patient_id=patient_id,
                medication_id=medication["id"],
                medication_name=medication["name"],
                rule_id=rule.id,
                rule_name=rule.name,
                rule_type=rule.type,
                severity=rule.severity,
                violation_description="Medication not covered by formulary",
                recommendation="Consider formulary alternatives",
                estimated_denial_probability=0.95,
                alternative_medications=med_status.get("alternatives", [])
            )
        
        return None
    
    def _load_payer_rules(self) -> List[PayerRule]:
        """Load payer rules from database."""
        # In production, this would load from a comprehensive rules database
        return [
            PayerRule(
                id="UHC_PA_001",
                name="Ozempic Prior Authorization",
                type=RuleType.PRIOR_AUTHORIZATION,
                description="Prior authorization required for Ozempic",
                criteria={
                    "required_rxcuis": ["1991302"],  # Ozempic
                    "pa_criteria": {
                        "required_diagnoses": ["E11.9", "E11.0"],  # Type 2 diabetes
                        "min_age": 18
                    },
                    "alternatives": ["Metformin", "Glipizide"]
                },
                action="require_pa",
                severity=RuleSeverity.ERROR,
                payer_id="UHC_001",
                effective_date=datetime(2024, 1, 1)
            ),
            PayerRule(
                id="MEDICARE_QL_001", 
                name="Opioid Quantity Limits",
                type=RuleType.QUANTITY_LIMIT,
                description="Quantity limits for opioid medications",
                criteria={
                    "quantity_limits": {
                        "1049621": {  # OxyContin
                            "max_quantity_per_fill": 30,
                            "max_days_supply": 7
                        }
                    }
                },
                action="limit_quantity",
                severity=RuleSeverity.WARNING,
                payer_id="MEDICARE_001",
                effective_date=datetime(2024, 1, 1)
            )
        ]
    
    def _load_formulary_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load formulary data by payer."""
        return {
            "UHC_001": {
                "1991302": {"status": "non_preferred", "tier": 3, "alternatives": ["Metformin"]},  # Ozempic
                "617311": {"status": "non_preferred", "tier": 3, "alternatives": ["Atorvastatin"]}  # Lipitor
            },
            "MEDICARE_001": {
                "1049621": {"status": "covered", "tier": 2, "alternatives": []}  # OxyContin
            }
        }
    
    def _load_step_therapy_protocols(self) -> Dict[str, List[str]]:
        """Load step therapy protocols."""
        return {
            "diabetes": ["Metformin", "Glipizide", "Ozempic"],
            "hypertension": ["Lisinopril", "Amlodipine", "ARBs"],
            "pain": ["Acetaminophen", "Ibuprofen", "Opioids"]
        }
    
    async def _analyze_coverage(self, patient_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall coverage patterns."""
        analysis = {
            "total_medications": 0,
            "covered_medications": 0,
            "preferred_medications": 0,
            "prior_auth_required": 0,
            "coverage_rate": 0.0
        }
        
        for patient_info in patient_data.values():
            payer_id = patient_info["insurance"]["payer_id"]
            formulary = self._formularies.get(payer_id, {})
            
            for medication in patient_info["medications"]:
                analysis["total_medications"] += 1
                rxcui = medication["rxcui"]
                med_status = formulary.get(rxcui, {})
                
                if med_status.get("status") != "not_covered":
                    analysis["covered_medications"] += 1
                
                if med_status.get("status") == "preferred":
                    analysis["preferred_medications"] += 1
        
        if analysis["total_medications"] > 0:
            analysis["coverage_rate"] = analysis["covered_medications"] / analysis["total_medications"]
        
        return analysis
    
    async def _store_rule_violations(self, violations: List[RuleViolation]):
        """Store rule violations to database."""
        logger.info(f"Storing {len(violations)} rule violations")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([
            {
                "patient_id": v.patient_id,
                "medication_id": v.medication_id,
                "medication_name": v.medication_name,
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "rule_type": v.rule_type.value,
                "severity": v.severity.value,
                "violation_description": v.violation_description,
                "recommendation": v.recommendation,
                "estimated_denial_probability": v.estimated_denial_probability,
                "alternative_medications": "; ".join(v.alternative_medications)
            }
            for v in violations
        ])
        
        # In a real implementation, this would use SQLAlchemy to store to database
        logger.info(f"Rule violations summary:\n{df.groupby(['rule_type', 'severity']).size()}")
    
    def get_rules_stats(self) -> Dict[str, Any]:
        """Get payer rules statistics."""
        return {
            "total_rules_loaded": len(self._payer_rules),
            "rules_by_type": {rule_type.value: len([r for r in self._payer_rules if r.type == rule_type]) for rule_type in RuleType},
            "payers_covered": len(set(rule.payer_id for rule in self._payer_rules)),
            "formulary_entries": sum(len(formulary) for formulary in self._formularies.values())
        }