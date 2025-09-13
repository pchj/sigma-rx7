"""Feature table export functionality for SigmaRx7."""

import asyncio
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import structlog
import json

logger = structlog.get_logger()


class FeatureExporter:
    """Feature table export engine for ML and analytics."""
    
    def __init__(self, config):
        """Initialize feature exporter."""
        self.config = config
        self.export_formats = ["csv", "parquet", "json"]
    
    async def export_features(self, output_dir: str = "output") -> Dict[str, Any]:
        """Export feature tables for ML and analytics."""
        logger.info(f"Starting feature table export to {output_dir}")
        
        result = {
            "output_directory": output_dir,
            "tables_exported": 0,
            "total_rows": 0,
            "export_files": [],
            "feature_tables": {}
        }
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export different feature tables
            feature_tables = [
                ("patient_features", self._export_patient_features),
                ("medication_features", self._export_medication_features),
                ("overlap_features", self._export_overlap_features),
                ("recommendation_features", self._export_recommendation_features),
                ("payer_rules_features", self._export_payer_rules_features),
                ("denial_prediction_features", self._export_denial_prediction_features),
                ("cost_analysis_features", self._export_cost_analysis_features)
            ]
            
            for table_name, export_func in feature_tables:
                try:
                    table_result = await export_func(output_path)
                    result["feature_tables"][table_name] = table_result
                    result["tables_exported"] += 1
                    result["total_rows"] += table_result.get("row_count", 0)
                    result["export_files"].extend(table_result.get("files", []))
                except Exception as e:
                    logger.error(f"Failed to export {table_name}: {e}")
                    result["feature_tables"][table_name] = {"error": str(e)}
            
            # Export feature metadata
            metadata_result = await self._export_feature_metadata(output_path, result)
            result["export_files"].extend(metadata_result.get("files", []))
            
            logger.info(f"Feature export completed: {result['tables_exported']} tables with {result['total_rows']} total rows")
        
        except Exception as e:
            logger.error(f"Feature export failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _export_patient_features(self, output_path: Path) -> Dict[str, Any]:
        """Export patient feature table."""
        logger.info("Exporting patient features")
        
        # Get patient data
        patient_data = await self._get_patient_feature_data()
        
        if patient_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "patient_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                patient_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                patient_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                patient_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(patient_data),
            "column_count": len(patient_data.columns),
            "files": files,
            "columns": list(patient_data.columns)
        }
    
    async def _export_medication_features(self, output_path: Path) -> Dict[str, Any]:
        """Export medication feature table."""
        logger.info("Exporting medication features")
        
        # Get medication data
        medication_data = await self._get_medication_feature_data()
        
        if medication_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "medication_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                medication_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                medication_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                medication_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(medication_data),
            "column_count": len(medication_data.columns),
            "files": files,
            "columns": list(medication_data.columns)
        }
    
    async def _export_overlap_features(self, output_path: Path) -> Dict[str, Any]:
        """Export overlap analysis feature table."""
        logger.info("Exporting overlap features")
        
        # Get overlap data
        overlap_data = await self._get_overlap_feature_data()
        
        if overlap_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "overlap_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                overlap_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                overlap_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                overlap_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(overlap_data),
            "column_count": len(overlap_data.columns),
            "files": files,
            "columns": list(overlap_data.columns)
        }
    
    async def _export_recommendation_features(self, output_path: Path) -> Dict[str, Any]:
        """Export generic recommendation feature table."""
        logger.info("Exporting recommendation features")
        
        # Get recommendation data
        recommendation_data = await self._get_recommendation_feature_data()
        
        if recommendation_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "recommendation_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                recommendation_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                recommendation_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                recommendation_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(recommendation_data),
            "column_count": len(recommendation_data.columns),
            "files": files,
            "columns": list(recommendation_data.columns)
        }
    
    async def _export_payer_rules_features(self, output_path: Path) -> Dict[str, Any]:
        """Export payer rules analysis feature table."""
        logger.info("Exporting payer rules features")
        
        # Get payer rules data
        payer_rules_data = await self._get_payer_rules_feature_data()
        
        if payer_rules_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "payer_rules_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                payer_rules_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                payer_rules_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                payer_rules_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(payer_rules_data),
            "column_count": len(payer_rules_data.columns),
            "files": files,
            "columns": list(payer_rules_data.columns)
        }
    
    async def _export_denial_prediction_features(self, output_path: Path) -> Dict[str, Any]:
        """Export denial prediction feature table."""
        logger.info("Exporting denial prediction features")
        
        # Get denial prediction data
        denial_data = await self._get_denial_prediction_feature_data()
        
        if denial_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "denial_prediction_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                denial_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                denial_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                denial_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(denial_data),
            "column_count": len(denial_data.columns),
            "files": files,
            "columns": list(denial_data.columns)
        }
    
    async def _export_cost_analysis_features(self, output_path: Path) -> Dict[str, Any]:
        """Export cost analysis feature table."""
        logger.info("Exporting cost analysis features")
        
        # Get cost analysis data
        cost_data = await self._get_cost_analysis_feature_data()
        
        if cost_data.empty:
            return {"row_count": 0, "files": []}
        
        # Export in multiple formats
        files = []
        base_filename = "cost_analysis_features"
        
        for format_type in self.export_formats:
            file_path = output_path / f"{base_filename}.{format_type}"
            
            if format_type == "csv":
                cost_data.to_csv(file_path, index=False)
            elif format_type == "parquet":
                cost_data.to_parquet(file_path, index=False)
            elif format_type == "json":
                cost_data.to_json(file_path, orient="records", indent=2)
            
            files.append(str(file_path))
        
        return {
            "row_count": len(cost_data),
            "column_count": len(cost_data.columns),
            "files": files,
            "columns": list(cost_data.columns)
        }
    
    async def _export_feature_metadata(self, output_path: Path, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Export feature metadata and data dictionary."""
        logger.info("Exporting feature metadata")
        
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "export_summary": {
                "total_tables": export_result["tables_exported"],
                "total_rows": export_result["total_rows"]
            },
            "table_metadata": {},
            "data_dictionary": await self._generate_data_dictionary()
        }
        
        # Add metadata for each table
        for table_name, table_info in export_result["feature_tables"].items():
            if "error" not in table_info:
                metadata["table_metadata"][table_name] = {
                    "row_count": table_info.get("row_count", 0),
                    "column_count": table_info.get("column_count", 0),
                    "columns": table_info.get("columns", [])
                }
        
        # Save metadata
        metadata_file = output_path / "feature_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data dictionary as CSV
        data_dict_df = pd.DataFrame(metadata["data_dictionary"])
        data_dict_file = output_path / "data_dictionary.csv"
        data_dict_df.to_csv(data_dict_file, index=False)
        
        return {
            "files": [str(metadata_file), str(data_dict_file)]
        }
    
    async def _get_patient_feature_data(self) -> pd.DataFrame:
        """Get patient feature data."""
        # In production, this would query the database
        # For demo purposes, generate sample data
        data = {
            "patient_id": ["patient_001", "patient_002", "patient_003"],
            "age": [45, 67, 72],
            "gender": ["M", "F", "F"],
            "total_medications": [2, 1, 1],
            "total_medication_cost": [470.0, 120.0, 45.0],
            "high_risk_medications": [1, 0, 1],
            "chronic_conditions_count": [2, 3, 2],
            "payer_type": ["commercial", "commercial", "medicare"],
            "formulary_compliance_rate": [0.5, 1.0, 1.0],
            "generic_utilization_rate": [0.0, 0.0, 0.0],
            "prior_auth_medications": [1, 0, 0],
            "step_therapy_violations": [1, 0, 0],
            "predicted_denial_risk": [0.85, 0.15, 0.65],
            "potential_cost_savings": [145.0, 95.0, 0.0]
        }
        
        return pd.DataFrame(data)
    
    async def _get_medication_feature_data(self) -> pd.DataFrame:
        """Get medication feature data."""
        data = {
            "medication_id": ["med_001", "med_002", "med_003"],
            "patient_id": ["patient_001", "patient_001", "patient_002"],
            "medication_name": ["Ozempic 0.5mg", "Lipitor 40mg", "OxyContin 20mg"],
            "rxcui": ["1991302", "617311", "1049621"],
            "rxnorm_name": ["Ozempic 0.5mg", "Atorvastatin 40mg", "OxyContin 20mg"],
            "therapeutic_class": ["diabetes", "statin", "opioid"],
            "is_brand": [True, True, True],
            "has_generic": [False, True, False],
            "monthly_cost": [350.0, 120.0, 45.0],
            "formulary_tier": [3, 3, 2],
            "prior_auth_required": [True, False, False],
            "step_therapy_required": [True, False, False],
            "quantity_prescribed": [1, 30, 60],
            "days_supply": [28, 30, 30],
            "quantity_ratio": [1.0, 1.0, 2.0],
            "overlap_detected": [False, False, False],
            "interaction_risk": ["low", "low", "high"],
            "denial_probability": [0.85, 0.25, 0.65],
            "recommended_action": ["prior_auth", "generic_switch", "quantity_limit"]
        }
        
        return pd.DataFrame(data)
    
    async def _get_overlap_feature_data(self) -> pd.DataFrame:
        """Get overlap analysis feature data."""
        data = {
            "overlap_id": ["overlap_001"],
            "patient_id": ["patient_001"],
            "medication1_id": ["med_001"],
            "medication2_id": ["med_002"],
            "overlap_type": ["potential_interaction"],
            "severity": ["low"],
            "overlap_days": [15],
            "confidence_score": [0.7],
            "recommendation": ["monitor_patient"]
        }
        
        return pd.DataFrame(data)
    
    async def _get_recommendation_feature_data(self) -> pd.DataFrame:
        """Get generic recommendation feature data."""
        data = {
            "recommendation_id": ["rec_001"],
            "patient_id": ["patient_001"],
            "brand_medication_id": ["med_002"],
            "brand_name": ["Lipitor 40mg"],
            "generic_name": ["Atorvastatin 40mg"],
            "monthly_cost_savings": [95.0],
            "annual_cost_savings": [1140.0],
            "therapeutic_equivalence": ["AB"],
            "formulary_improvement": [True],
            "confidence_score": [0.9]
        }
        
        return pd.DataFrame(data)
    
    async def _get_payer_rules_feature_data(self) -> pd.DataFrame:
        """Get payer rules analysis feature data."""
        data = {
            "violation_id": ["violation_001"],
            "patient_id": ["patient_001"],
            "medication_id": ["med_001"],
            "rule_type": ["prior_authorization"],
            "severity": ["error"],
            "estimated_denial_probability": [0.8],
            "payer_id": ["UHC_001"],
            "rule_name": ["Ozempic Prior Authorization"]
        }
        
        return pd.DataFrame(data)
    
    async def _get_denial_prediction_feature_data(self) -> pd.DataFrame:
        """Get denial prediction feature data."""
        data = {
            "prediction_id": ["pred_001", "pred_002", "pred_003"],
            "patient_id": ["patient_001", "patient_001", "patient_002"],
            "medication_id": ["med_001", "med_002", "med_003"],
            "denial_probability": [0.85, 0.25, 0.65],
            "risk_category": ["high", "low", "medium"],
            "contributing_factors": [
                "High medication cost; Prior authorization required",
                "Non-preferred formulary tier",
                "Excessive quantity prescribed"
            ],
            "confidence_score": [0.8, 0.8, 0.8],
            "model_version": ["1.0", "1.0", "1.0"]
        }
        
        return pd.DataFrame(data)
    
    async def _get_cost_analysis_feature_data(self) -> pd.DataFrame:
        """Get cost analysis feature data."""
        data = {
            "analysis_id": ["cost_001", "cost_002"],
            "patient_id": ["patient_001", "patient_001"],
            "medication_id": ["med_002", "med_001"],
            "current_monthly_cost": [120.0, 350.0],
            "alternative_monthly_cost": [25.0, 350.0],
            "monthly_savings": [95.0, 0.0],
            "annual_savings": [1140.0, 0.0],
            "patient_copay_current": [36.0, 105.0],
            "patient_copay_alternative": [2.5, 105.0],
            "patient_savings": [33.5, 0.0],
            "savings_percentage": [79.2, 0.0]
        }
        
        return pd.DataFrame(data)
    
    async def _generate_data_dictionary(self) -> List[Dict[str, str]]:
        """Generate data dictionary for all feature tables."""
        data_dictionary = [
            # Patient Features
            {"table": "patient_features", "column": "patient_id", "type": "string", "description": "Unique patient identifier"},
            {"table": "patient_features", "column": "age", "type": "integer", "description": "Patient age in years"},
            {"table": "patient_features", "column": "gender", "type": "string", "description": "Patient gender (M/F)"},
            {"table": "patient_features", "column": "total_medications", "type": "integer", "description": "Total number of medications"},
            {"table": "patient_features", "column": "total_medication_cost", "type": "float", "description": "Total monthly medication cost in USD"},
            {"table": "patient_features", "column": "high_risk_medications", "type": "integer", "description": "Number of high-risk medications"},
            {"table": "patient_features", "column": "chronic_conditions_count", "type": "integer", "description": "Number of chronic conditions"},
            {"table": "patient_features", "column": "payer_type", "type": "string", "description": "Insurance payer type"},
            {"table": "patient_features", "column": "formulary_compliance_rate", "type": "float", "description": "Formulary compliance rate (0-1)"},
            {"table": "patient_features", "column": "generic_utilization_rate", "type": "float", "description": "Generic medication utilization rate (0-1)"},
            {"table": "patient_features", "column": "prior_auth_medications", "type": "integer", "description": "Number of medications requiring prior authorization"},
            {"table": "patient_features", "column": "step_therapy_violations", "type": "integer", "description": "Number of step therapy violations"},
            {"table": "patient_features", "column": "predicted_denial_risk", "type": "float", "description": "ML-predicted denial risk score (0-1)"},
            {"table": "patient_features", "column": "potential_cost_savings", "type": "float", "description": "Potential monthly cost savings in USD"},
            
            # Medication Features
            {"table": "medication_features", "column": "medication_id", "type": "string", "description": "Unique medication identifier"},
            {"table": "medication_features", "column": "patient_id", "type": "string", "description": "Associated patient identifier"},
            {"table": "medication_features", "column": "medication_name", "type": "string", "description": "Medication name"},
            {"table": "medication_features", "column": "rxcui", "type": "string", "description": "RxNorm concept identifier"},
            {"table": "medication_features", "column": "rxnorm_name", "type": "string", "description": "Normalized RxNorm name"},
            {"table": "medication_features", "column": "therapeutic_class", "type": "string", "description": "Therapeutic classification"},
            {"table": "medication_features", "column": "is_brand", "type": "boolean", "description": "Whether medication is brand name"},
            {"table": "medication_features", "column": "has_generic", "type": "boolean", "description": "Whether generic alternative exists"},
            {"table": "medication_features", "column": "monthly_cost", "type": "float", "description": "Monthly medication cost in USD"},
            {"table": "medication_features", "column": "formulary_tier", "type": "integer", "description": "Formulary tier (1-4)"},
            {"table": "medication_features", "column": "prior_auth_required", "type": "boolean", "description": "Whether prior authorization is required"},
            {"table": "medication_features", "column": "step_therapy_required", "type": "boolean", "description": "Whether step therapy is required"},
            {"table": "medication_features", "column": "quantity_prescribed", "type": "integer", "description": "Quantity prescribed"},
            {"table": "medication_features", "column": "days_supply", "type": "integer", "description": "Days supply"},
            {"table": "medication_features", "column": "quantity_ratio", "type": "float", "description": "Ratio of prescribed to standard quantity"},
            {"table": "medication_features", "column": "overlap_detected", "type": "boolean", "description": "Whether medication overlap was detected"},
            {"table": "medication_features", "column": "interaction_risk", "type": "string", "description": "Drug interaction risk level"},
            {"table": "medication_features", "column": "denial_probability", "type": "float", "description": "Predicted denial probability (0-1)"},
            {"table": "medication_features", "column": "recommended_action", "type": "string", "description": "Recommended action to reduce denial risk"},
        ]
        
        return data_dictionary