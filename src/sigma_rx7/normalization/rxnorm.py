"""RxNorm medication normalization for SigmaRx7."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
import structlog
import pandas as pd
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class RxNormConcept:
    """RxNorm concept representation."""
    rxcui: str
    name: str
    synonym: str
    tty: str  # Term Type
    language: str = "ENG"
    suppress: str = "N"


@dataclass
class MedicationMapping:
    """Medication to RxNorm mapping."""
    original_name: str
    rxcui: Optional[str]
    rxnorm_name: Optional[str]
    ingredient: Optional[str]
    strength: Optional[str]
    dose_form: Optional[str]
    confidence_score: float
    mapping_method: str


class RxNormNormalizer:
    """RxNorm medication normalization engine."""
    
    def __init__(self, config):
        """Initialize RxNorm normalizer."""
        self.config = config
        self.api_url = config.rxnorm.api_url
        self.timeout = config.rxnorm.timeout
        self.rate_limit = config.rxnorm.rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SigmaRx7-Pipeline/1.0',
            'Accept': 'application/json'
        })
        self._concept_cache = {}
    
    async def normalize_medications(self) -> Dict[str, Any]:
        """Normalize all medications in the database using RxNorm."""
        logger.info("Starting RxNorm medication normalization")
        
        result = {
            "total_medications": 0,
            "normalized_count": 0,
            "failed_count": 0,
            "mappings": [],
            "errors": []
        }
        
        try:
            # Get medications from database
            medications = await self._get_medications_from_db()
            result["total_medications"] = len(medications)
            
            # Process medications in batches
            batch_size = 50
            for i in range(0, len(medications), batch_size):
                batch = medications[i:i + batch_size]
                batch_results = await self._normalize_medication_batch(batch)
                
                result["mappings"].extend(batch_results["mappings"])
                result["normalized_count"] += batch_results["normalized_count"]
                result["failed_count"] += batch_results["failed_count"]
                result["errors"].extend(batch_results["errors"])
                
                # Rate limiting
                if self.rate_limit > 0:
                    await asyncio.sleep(self.rate_limit)
            
            # Store normalized results back to database
            await self._store_normalized_medications(result["mappings"])
            
            logger.info(f"RxNorm normalization completed: {result['normalized_count']}/{result['total_medications']} medications normalized")
        
        except Exception as e:
            logger.error(f"RxNorm normalization failed: {e}")
            result["errors"].append(str(e))
        
        return result
    
    async def _get_medications_from_db(self) -> List[Dict[str, Any]]:
        """Get medications from database for normalization."""
        # Placeholder - would query actual database
        # For demo purposes, return sample medications
        return [
            {"id": 1, "name": "Tylenol 500mg", "generic_name": "acetaminophen"},
            {"id": 2, "name": "Advil 200mg", "generic_name": "ibuprofen"},
            {"id": 3, "name": "Lipitor 20mg", "generic_name": "atorvastatin"},
            {"id": 4, "name": "Metformin 500mg", "generic_name": "metformin"},
            {"id": 5, "name": "Lisinopril 10mg", "generic_name": "lisinopril"}
        ]
    
    async def _normalize_medication_batch(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize a batch of medications."""
        batch_result = {
            "normalized_count": 0,
            "failed_count": 0,
            "mappings": [],
            "errors": []
        }
        
        for medication in medications:
            try:
                mapping = await self._normalize_single_medication(medication)
                batch_result["mappings"].append(mapping)
                
                if mapping.rxcui:
                    batch_result["normalized_count"] += 1
                else:
                    batch_result["failed_count"] += 1
            
            except Exception as e:
                logger.error(f"Error normalizing medication {medication.get('name', 'Unknown')}: {e}")
                batch_result["failed_count"] += 1
                batch_result["errors"].append(f"Medication {medication.get('id')}: {str(e)}")
        
        return batch_result
    
    async def _normalize_single_medication(self, medication: Dict[str, Any]) -> MedicationMapping:
        """Normalize a single medication to RxNorm."""
        med_name = medication.get("name", "")
        generic_name = medication.get("generic_name", "")
        
        logger.debug(f"Normalizing medication: {med_name}")
        
        # Try multiple normalization strategies
        strategies = [
            ("exact_match", lambda: self._exact_match_search(med_name)),
            ("approximate_match", lambda: self._approximate_match_search(med_name)),
            ("generic_search", lambda: self._exact_match_search(generic_name) if generic_name else None),
            ("spelling_suggestion", lambda: self._spelling_suggestion_search(med_name))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                rxnorm_result = await strategy_func()
                if rxnorm_result:
                    return MedicationMapping(
                        original_name=med_name,
                        rxcui=rxnorm_result.get("rxcui"),
                        rxnorm_name=rxnorm_result.get("name"),
                        ingredient=rxnorm_result.get("ingredient"),
                        strength=rxnorm_result.get("strength"),
                        dose_form=rxnorm_result.get("dose_form"),
                        confidence_score=rxnorm_result.get("confidence", 0.0),
                        mapping_method=strategy_name
                    )
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed for {med_name}: {e}")
                continue
        
        # No mapping found
        return MedicationMapping(
            original_name=med_name,
            rxcui=None,
            rxnorm_name=None,
            ingredient=None,
            strength=None,
            dose_form=None,
            confidence_score=0.0,
            mapping_method="no_match"
        )
    
    async def _exact_match_search(self, term: str) -> Optional[Dict[str, Any]]:
        """Perform exact match search in RxNorm."""
        if not term:
            return None
        
        # Check cache first
        cache_key = f"exact_{term.lower()}"
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        
        try:
            url = f"{self.api_url}/rxcui.json"
            params = {"name": term, "search": "1"}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
            
            if rxcui_list:
                rxcui = rxcui_list[0]  # Take first match
                # Get detailed concept information
                concept_info = await self._get_concept_details(rxcui)
                
                result = {
                    "rxcui": rxcui,
                    "name": concept_info.get("name", term),
                    "confidence": 1.0,
                    **concept_info
                }
                
                self._concept_cache[cache_key] = result
                return result
        
        except Exception as e:
            logger.debug(f"Exact match search failed for '{term}': {e}")
        
        return None
    
    async def _approximate_match_search(self, term: str) -> Optional[Dict[str, Any]]:
        """Perform approximate match search in RxNorm."""
        if not term:
            return None
        
        cache_key = f"approx_{term.lower()}"
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        
        try:
            url = f"{self.api_url}/approximateTerm.json"
            params = {"term": term, "maxEntries": "5"}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            
            if candidates:
                # Take the first (best) candidate
                candidate = candidates[0] if isinstance(candidates, list) else candidates
                rxcui = candidate.get("rxcui")
                score = float(candidate.get("score", 0))
                
                if rxcui and score > 0.7:  # Confidence threshold
                    concept_info = await self._get_concept_details(rxcui)
                    
                    result = {
                        "rxcui": rxcui,
                        "name": concept_info.get("name", term),
                        "confidence": score,
                        **concept_info
                    }
                    
                    self._concept_cache[cache_key] = result
                    return result
        
        except Exception as e:
            logger.debug(f"Approximate match search failed for '{term}': {e}")
        
        return None
    
    async def _spelling_suggestion_search(self, term: str) -> Optional[Dict[str, Any]]:
        """Perform spelling suggestion search in RxNorm."""
        if not term:
            return None
        
        try:
            url = f"{self.api_url}/spellingsuggestions.json"
            params = {"name": term}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            suggestions = data.get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion", [])
            
            if suggestions:
                # Try the first suggestion
                suggestion = suggestions[0] if isinstance(suggestions, list) else suggestions
                return await self._exact_match_search(suggestion)
        
        except Exception as e:
            logger.debug(f"Spelling suggestion search failed for '{term}': {e}")
        
        return None
    
    async def _get_concept_details(self, rxcui: str) -> Dict[str, Any]:
        """Get detailed information about an RxNorm concept."""
        cache_key = f"concept_{rxcui}"
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        
        details = {}
        
        try:
            # Get concept properties
            url = f"{self.api_url}/rxcui/{rxcui}/properties.json"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            properties = data.get("properties", {})
            
            details.update({
                "name": properties.get("name", ""),
                "synonym": properties.get("synonym", ""),
                "tty": properties.get("tty", ""),  # Term Type
            })
            
            # Get related concepts (ingredients, strength, etc.)
            related_url = f"{self.api_url}/rxcui/{rxcui}/related.json"
            related_response = self.session.get(related_url, timeout=self.timeout)
            related_response.raise_for_status()
            
            related_data = related_response.json()
            concept_group = related_data.get("relatedGroup", {}).get("conceptGroup", [])
            
            # Extract ingredient, strength, dose form
            for group in concept_group:
                tty = group.get("tty")
                concept_properties = group.get("conceptProperties", [])
                
                if tty == "IN" and concept_properties:  # Ingredient
                    details["ingredient"] = concept_properties[0].get("name")
                elif tty == "SCDF" and concept_properties:  # Dose form
                    details["dose_form"] = concept_properties[0].get("name")
            
            # Extract strength from name if not found in related concepts
            if "strength" not in details:
                details["strength"] = self._extract_strength_from_name(details.get("name", ""))
            
            self._concept_cache[cache_key] = details
        
        except Exception as e:
            logger.debug(f"Error getting concept details for {rxcui}: {e}")
        
        return details
    
    def _extract_strength_from_name(self, name: str) -> Optional[str]:
        """Extract strength information from medication name."""
        import re
        
        # Common strength patterns
        strength_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|units?|iu|mEq)',
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg)',
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg)'
        ]
        
        for pattern in strength_patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    async def _store_normalized_medications(self, mappings: List[MedicationMapping]):
        """Store normalized medication mappings to database."""
        logger.info(f"Storing {len(mappings)} medication mappings to database")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([
            {
                "original_name": m.original_name,
                "rxcui": m.rxcui,
                "rxnorm_name": m.rxnorm_name,
                "ingredient": m.ingredient,
                "strength": m.strength,
                "dose_form": m.dose_form,
                "confidence_score": m.confidence_score,
                "mapping_method": m.mapping_method
            }
            for m in mappings
        ])
        
        # In a real implementation, this would use SQLAlchemy to store to database
        # For now, just log the results
        logger.info(f"Normalized medications summary:\n{df.groupby('mapping_method').size()}")
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return {
            "cache_size": len(self._concept_cache),
            "api_url": self.api_url,
            "rate_limit": self.rate_limit
        }