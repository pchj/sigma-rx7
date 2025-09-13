"""Generic medication recommendations for SigmaRx7."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class GenericRecommendation:
    """Generic medication recommendation."""
    patient_id: str
    brand_medication_id: str
    brand_name: str
    brand_rxcui: str
    generic_name: str
    generic_rxcui: str
    cost_savings_monthly: float
    cost_savings_annual: float
    therapeutic_equivalence: str  # AB, AA, etc.
    recommendation_reason: str
    confidence_score: float
    formulary_status: str  # preferred, non_preferred, not_covered


@dataclass
class CostAnalysis:
    """Cost analysis for medication substitution."""
    brand_cost: float
    generic_cost: float
    savings_amount: float
    savings_percentage: float
    patient_copay_brand: float
    patient_copay_generic: float


class GenericRecommender:
    """Generic medication recommendation engine."""
    
    def __init__(self, config):
        """Initialize generic recommender."""
        self.config = config
        self._generic_mappings = self._load_generic_mappings()
        self._cost_data = self._load_cost_data()
        self._formulary_data = self._load_formulary_data()
    
    async def recommend_generics(self) -> Dict[str, Any]:
        """Generate generic medication recommendations for all patients."""
        logger.info("Starting generic medication recommendations")
        
        result = {
            "total_patients": 0,
            "patients_with_recommendations": 0,
            "total_recommendations": 0,
            "potential_monthly_savings": 0.0,
            "potential_annual_savings": 0.0,
            "recommendations": [],
            "savings_by_therapeutic_class": {},
            "formulary_alignment": {"improved": 0, "maintained": 0, "worsened": 0}
        }
        
        try:
            # Get patient medication data
            patient_medications = await self._get_patient_medications()
            result["total_patients"] = len(patient_medications)
            
            # Process each patient
            for patient_id, medications in patient_medications.items():
                patient_recommendations = await self._generate_patient_recommendations(patient_id, medications)
                
                if patient_recommendations:
                    result["patients_with_recommendations"] += 1
                    result["recommendations"].extend(patient_recommendations)
                    
                    # Calculate savings
                    for rec in patient_recommendations:
                        result["total_recommendations"] += 1
                        result["potential_monthly_savings"] += rec.cost_savings_monthly
                        result["potential_annual_savings"] += rec.cost_savings_annual
                        
                        # Track savings by therapeutic class
                        therapeutic_class = self._get_therapeutic_class(rec.brand_rxcui)
                        if therapeutic_class not in result["savings_by_therapeutic_class"]:
                            result["savings_by_therapeutic_class"][therapeutic_class] = 0.0
                        result["savings_by_therapeutic_class"][therapeutic_class] += rec.cost_savings_annual
                        
                        # Track formulary alignment
                        if rec.formulary_status == "preferred":
                            result["formulary_alignment"]["improved"] += 1
                        elif rec.formulary_status == "non_preferred":
                            result["formulary_alignment"]["maintained"] += 1
                        else:
                            result["formulary_alignment"]["worsened"] += 1
            
            # Store recommendations
            await self._store_recommendations(result["recommendations"])
            
            logger.info(f"Generic recommendations completed: {result['total_recommendations']} recommendations with ${result['potential_annual_savings']:.2f} annual savings potential")
        
        except Exception as e:
            logger.error(f"Generic recommendation failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _get_patient_medications(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get medication data for all patients."""
        # Placeholder - would query actual database
        # For demo purposes, return sample data with brand medications
        return {
            "patient_001": [
                {
                    "id": "med_001",
                    "name": "Lipitor 20mg",
                    "rxcui": "617314",
                    "is_brand": True,
                    "therapeutic_class": "statin",
                    "daily_dose": "20mg once daily",
                    "quantity": 30,
                    "days_supply": 30
                },
                {
                    "id": "med_002",
                    "name": "Advil 200mg",
                    "rxcui": "5640",
                    "is_brand": True,
                    "therapeutic_class": "nsaid",
                    "daily_dose": "200mg as needed",
                    "quantity": 60,
                    "days_supply": 30
                }
            ],
            "patient_002": [
                {
                    "id": "med_003",
                    "name": "Prinivil 10mg",
                    "rxcui": "206765",
                    "is_brand": True,
                    "therapeutic_class": "ace_inhibitor",
                    "daily_dose": "10mg once daily",
                    "quantity": 30,
                    "days_supply": 30
                },
                {
                    "id": "med_004",
                    "name": "Atorvastatin 20mg",  # Already generic
                    "rxcui": "861634",
                    "is_brand": False,
                    "therapeutic_class": "statin",
                    "daily_dose": "20mg once daily",
                    "quantity": 30,
                    "days_supply": 30
                }
            ]
        }
    
    async def _generate_patient_recommendations(self, patient_id: str, medications: List[Dict[str, Any]]) -> List[GenericRecommendation]:
        """Generate generic recommendations for a single patient."""
        recommendations = []
        
        for medication in medications:
            # Only recommend generics for brand medications
            if not medication.get("is_brand", False):
                continue
            
            brand_rxcui = medication["rxcui"]
            
            # Find generic alternative
            generic_alternative = await self._find_generic_alternative(brand_rxcui)
            if not generic_alternative:
                continue
            
            # Perform cost analysis
            cost_analysis = await self._analyze_costs(brand_rxcui, generic_alternative["rxcui"], medication)
            if cost_analysis.savings_amount <= 0:
                continue  # No cost savings
            
            # Check formulary status
            formulary_status = await self._check_formulary_status(generic_alternative["rxcui"])
            
            # Create recommendation
            recommendation = GenericRecommendation(
                patient_id=patient_id,
                brand_medication_id=medication["id"],
                brand_name=medication["name"],
                brand_rxcui=brand_rxcui,
                generic_name=generic_alternative["name"],
                generic_rxcui=generic_alternative["rxcui"],
                cost_savings_monthly=cost_analysis.savings_amount,
                cost_savings_annual=cost_analysis.savings_amount * 12,
                therapeutic_equivalence=generic_alternative.get("therapeutic_equivalence", "AB"),
                recommendation_reason=self._generate_recommendation_reason(cost_analysis, formulary_status),
                confidence_score=self._calculate_confidence_score(generic_alternative, cost_analysis, formulary_status),
                formulary_status=formulary_status
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _find_generic_alternative(self, brand_rxcui: str) -> Optional[Dict[str, Any]]:
        """Find generic alternative for a brand medication."""
        # Check generic mappings
        if brand_rxcui in self._generic_mappings:
            return self._generic_mappings[brand_rxcui]
        
        # If not in mappings, try to find via RxNorm API (simplified)
        # In production, this would use the actual RxNorm API
        brand_to_generic_mappings = {
            "617314": {  # Lipitor -> Atorvastatin
                "rxcui": "861634",
                "name": "Atorvastatin 20mg",
                "therapeutic_equivalence": "AB"
            },
            "5640": {  # Advil -> Ibuprofen
                "rxcui": "5690",
                "name": "Ibuprofen 200mg",
                "therapeutic_equivalence": "AB"
            },
            "206765": {  # Prinivil -> Lisinopril
                "rxcui": "29046",
                "name": "Lisinopril 10mg",
                "therapeutic_equivalence": "AB"
            }
        }
        
        return brand_to_generic_mappings.get(brand_rxcui)
    
    async def _analyze_costs(self, brand_rxcui: str, generic_rxcui: str, medication: Dict[str, Any]) -> CostAnalysis:
        """Analyze cost differences between brand and generic."""
        # Get cost data
        brand_cost = self._cost_data.get(brand_rxcui, {}).get("awp", 100.0)  # Default AWP
        generic_cost = self._cost_data.get(generic_rxcui, {}).get("awp", 30.0)  # Default generic AWP
        
        # Calculate monthly cost based on quantity
        quantity = medication.get("quantity", 30)
        days_supply = medication.get("days_supply", 30)
        monthly_multiplier = 30 / days_supply
        
        monthly_brand_cost = brand_cost * monthly_multiplier
        monthly_generic_cost = generic_cost * monthly_multiplier
        
        savings_amount = monthly_brand_cost - monthly_generic_cost
        savings_percentage = (savings_amount / monthly_brand_cost) * 100 if monthly_brand_cost > 0 else 0
        
        # Calculate patient copays (simplified)
        # In production, this would use actual insurance benefit information
        brand_copay = min(monthly_brand_cost * 0.3, 50.0)  # 30% coinsurance, max $50
        generic_copay = min(monthly_generic_cost * 0.1, 10.0)  # 10% coinsurance, max $10
        
        return CostAnalysis(
            brand_cost=monthly_brand_cost,
            generic_cost=monthly_generic_cost,
            savings_amount=savings_amount,
            savings_percentage=savings_percentage,
            patient_copay_brand=brand_copay,
            patient_copay_generic=generic_copay
        )
    
    async def _check_formulary_status(self, generic_rxcui: str) -> str:
        """Check formulary status of generic medication."""
        formulary_info = self._formulary_data.get(generic_rxcui, {})
        return formulary_info.get("status", "non_preferred")
    
    def _generate_recommendation_reason(self, cost_analysis: CostAnalysis, formulary_status: str) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        if cost_analysis.savings_percentage > 50:
            reasons.append(f"Significant cost savings: {cost_analysis.savings_percentage:.1f}%")
        elif cost_analysis.savings_percentage > 20:
            reasons.append(f"Moderate cost savings: {cost_analysis.savings_percentage:.1f}%")
        else:
            reasons.append(f"Cost savings: {cost_analysis.savings_percentage:.1f}%")
        
        if formulary_status == "preferred":
            reasons.append("Preferred on formulary")
        elif formulary_status == "non_preferred":
            reasons.append("Non-preferred on formulary")
        
        patient_savings = cost_analysis.patient_copay_brand - cost_analysis.patient_copay_generic
        if patient_savings > 0:
            reasons.append(f"Patient copay savings: ${patient_savings:.2f}/month")
        
        return "; ".join(reasons)
    
    def _calculate_confidence_score(self, generic_alternative: Dict[str, Any], cost_analysis: CostAnalysis, formulary_status: str) -> float:
        """Calculate confidence score for recommendation."""
        score = 0.0
        
        # Therapeutic equivalence
        te = generic_alternative.get("therapeutic_equivalence", "")
        if te == "AB":
            score += 0.4  # Highest therapeutic equivalence
        elif te == "AA":
            score += 0.35
        else:
            score += 0.2
        
        # Cost savings
        if cost_analysis.savings_percentage > 50:
            score += 0.3
        elif cost_analysis.savings_percentage > 20:
            score += 0.2
        else:
            score += 0.1
        
        # Formulary status
        if formulary_status == "preferred":
            score += 0.3
        elif formulary_status == "non_preferred":
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_therapeutic_class(self, rxcui: str) -> str:
        """Get therapeutic class for an RxCUI."""
        # Simplified mapping
        class_mappings = {
            "617314": "statin",  # Lipitor
            "861634": "statin",  # Atorvastatin
            "5640": "nsaid",     # Advil
            "5690": "nsaid",     # Ibuprofen
            "206765": "ace_inhibitor",  # Prinivil
            "29046": "ace_inhibitor"    # Lisinopril
        }
        return class_mappings.get(rxcui, "unknown")
    
    def _load_generic_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load brand to generic mappings."""
        # In production, this would load from a comprehensive database
        return {
            "617314": {  # Lipitor
                "rxcui": "861634",
                "name": "Atorvastatin 20mg",
                "therapeutic_equivalence": "AB"
            },
            "5640": {  # Advil
                "rxcui": "5690", 
                "name": "Ibuprofen 200mg",
                "therapeutic_equivalence": "AB"
            }
        }
    
    def _load_cost_data(self) -> Dict[str, Dict[str, float]]:
        """Load medication cost data."""
        # In production, this would load from pharmacy cost databases
        return {
            "617314": {"awp": 120.00, "wac": 100.00},  # Lipitor
            "861634": {"awp": 25.00, "wac": 20.00},    # Atorvastatin
            "5640": {"awp": 15.00, "wac": 12.00},      # Advil
            "5690": {"awp": 3.00, "wac": 2.50},       # Ibuprofen
            "206765": {"awp": 45.00, "wac": 35.00},    # Prinivil
            "29046": {"awp": 8.00, "wac": 6.00}       # Lisinopril
        }
    
    def _load_formulary_data(self) -> Dict[str, Dict[str, Any]]:
        """Load formulary status data."""
        # In production, this would load from payer formulary databases
        return {
            "861634": {"status": "preferred", "tier": 1},   # Atorvastatin
            "5690": {"status": "preferred", "tier": 1},     # Ibuprofen
            "29046": {"status": "preferred", "tier": 1},    # Lisinopril
            "617314": {"status": "non_preferred", "tier": 3}, # Lipitor
            "5640": {"status": "non_preferred", "tier": 2},   # Advil
            "206765": {"status": "non_preferred", "tier": 3}  # Prinivil
        }
    
    async def _store_recommendations(self, recommendations: List[GenericRecommendation]):
        """Store generic recommendations to database."""
        logger.info(f"Storing {len(recommendations)} generic recommendations")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([
            {
                "patient_id": r.patient_id,
                "brand_medication_id": r.brand_medication_id,
                "brand_name": r.brand_name,
                "brand_rxcui": r.brand_rxcui,
                "generic_name": r.generic_name,
                "generic_rxcui": r.generic_rxcui,
                "cost_savings_monthly": r.cost_savings_monthly,
                "cost_savings_annual": r.cost_savings_annual,
                "therapeutic_equivalence": r.therapeutic_equivalence,
                "recommendation_reason": r.recommendation_reason,
                "confidence_score": r.confidence_score,
                "formulary_status": r.formulary_status
            }
            for r in recommendations
        ])
        
        # In a real implementation, this would use SQLAlchemy to store to database
        logger.info(f"Generic recommendations summary:\n{df.groupby('formulary_status')['cost_savings_annual'].sum()}")
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get generic recommendation statistics."""
        return {
            "generic_mappings_loaded": len(self._generic_mappings),
            "cost_data_entries": len(self._cost_data),
            "formulary_entries": len(self._formulary_data)
        }