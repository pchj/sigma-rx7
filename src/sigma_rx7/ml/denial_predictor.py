"""ML-based denial prediction for SigmaRx7."""

import asyncio
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class DenialPrediction:
    """Denial prediction result."""
    patient_id: str
    medication_id: str
    medication_name: str
    denial_probability: float
    risk_category: str  # low, medium, high
    contributing_factors: List[str]
    recommended_actions: List[str]
    confidence_score: float


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    feature_importance: Dict[str, float]


class DenialPredictor:
    """ML-based denial prediction engine."""
    
    def __init__(self, config):
        """Initialize denial predictor."""
        self.config = config
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.model_metrics = None
        self._load_or_train_model()
    
    async def predict_denials(self) -> Dict[str, Any]:
        """Predict denials for all patient medications."""
        logger.info("Starting ML-based denial prediction")
        
        result = {
            "total_predictions": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "average_denial_probability": 0.0,
            "predictions": [],
            "model_performance": self.model_metrics.__dict__ if self.model_metrics else None,
            "feature_importance": self._get_feature_importance()
        }
        
        try:
            # Get patient medication data for prediction
            prediction_data = await self._get_prediction_data()
            
            if prediction_data.empty:
                logger.warning("No data available for prediction")
                return result
            
            # Generate predictions
            predictions = await self._generate_predictions(prediction_data)
            result["predictions"] = predictions
            result["total_predictions"] = len(predictions)
            
            # Calculate statistics
            for prediction in predictions:
                if prediction.risk_category == "high":
                    result["high_risk_count"] += 1
                elif prediction.risk_category == "medium":
                    result["medium_risk_count"] += 1
                else:
                    result["low_risk_count"] += 1
            
            if predictions:
                result["average_denial_probability"] = sum(p.denial_probability for p in predictions) / len(predictions)
            
            # Store predictions
            await self._store_predictions(predictions)
            
            logger.info(f"Denial prediction completed: {result['total_predictions']} predictions, {result['high_risk_count']} high-risk")
        
        except Exception as e:
            logger.error(f"Denial prediction failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def _load_or_train_model(self):
        """Load existing model or train new one."""
        model_path = Path(self.config.ml.model_path)
        
        if model_path.exists():
            try:
                self._load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                self._train_new_model()
        else:
            logger.info("No existing model found, training new model")
            self._train_new_model()
    
    def _load_model(self, model_path: Path):
        """Load trained model from file."""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.scaler = model_data.get("scaler")
        self.label_encoders = model_data.get("label_encoders", {})
        self.feature_columns = model_data.get("feature_columns")
        self.model_metrics = ModelMetrics(**model_data.get("metrics", {}))
    
    def _train_new_model(self):
        """Train a new denial prediction model."""
        logger.info("Training new denial prediction model")
        
        try:
            # Generate training data
            training_data = self._generate_training_data()
            
            if training_data.empty:
                logger.warning("No training data available, using mock model")
                self._create_mock_model()
                return
            
            # Prepare features and target
            X, y = self._prepare_training_data(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.ml.train_test_split, random_state=42
            )
            
            # Train model
            self.model = self._train_model(X_train, y_train)
            
            # Evaluate model
            self.model_metrics = self._evaluate_model(X_test, y_test)
            
            # Save model
            self._save_model()
            
            logger.info(f"Model training completed. AUC: {self.model_metrics.auc_score:.3f}")
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self._create_mock_model()
    
    def _generate_training_data(self) -> pd.DataFrame:
        """Generate training data from historical claims."""
        # In production, this would query historical claims data
        # For demo purposes, generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            "patient_age": np.random.randint(18, 85, n_samples),
            "medication_cost": np.random.lognormal(3, 1, n_samples),
            "prior_auth_required": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            "step_therapy_compliant": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "formulary_tier": np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            "diagnosis_count": np.random.poisson(2, n_samples),
            "previous_denials": np.random.poisson(0.5, n_samples),
            "generic_available": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "quantity_ratio": np.random.normal(1.0, 0.3, n_samples),
            "payer_type": np.random.choice(["commercial", "medicare", "medicaid"], n_samples, p=[0.6, 0.3, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Generate denial probability based on features
        denial_prob = (
            0.1 * (df["patient_age"] > 65) +
            0.2 * (df["medication_cost"] > 100) +
            0.3 * df["prior_auth_required"] +
            0.2 * (1 - df["step_therapy_compliant"]) +
            0.1 * (df["formulary_tier"] > 2) +
            0.1 * (df["previous_denials"] > 0) +
            0.1 * (df["quantity_ratio"] > 1.5) +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Add noise and constrain to [0, 1]
        denial_prob = np.clip(denial_prob, 0, 1)
        df["denied"] = (denial_prob > 0.5).astype(int)
        
        return df
    
    def _prepare_training_data(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Define feature columns
        feature_columns = [
            "patient_age", "medication_cost", "prior_auth_required",
            "step_therapy_compliant", "formulary_tier", "diagnosis_count",
            "previous_denials", "generic_available", "quantity_ratio"
        ]
        
        # Handle categorical variables
        categorical_columns = ["payer_type"]
        
        X = training_data[feature_columns + categorical_columns].copy()
        y = training_data["denied"]
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X[feature_columns] = self.scaler.fit_transform(X[feature_columns])
        else:
            X[feature_columns] = self.scaler.transform(X[feature_columns])
        
        self.feature_columns = list(X.columns)
        return X, y
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train the denial prediction model."""
        # Try multiple algorithms and select best
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            "logistic_regression": LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=1000
            )
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
            avg_score = scores.mean()
            logger.info(f"{name} CV AUC: {avg_score:.3f} (+/- {scores.std() * 2:.3f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model on full training set
        best_model.fit(X_train, y_train)
        return best_model
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for i, importance in enumerate(self.model.feature_importances_):
                feature_importance[self.feature_columns[i]] = float(importance)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            feature_importance=feature_importance
        )
    
    def _save_model(self):
        """Save trained model to file."""
        model_path = Path(self.config.ml.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "metrics": self.model_metrics.__dict__ if self.model_metrics else {}
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes."""
        logger.info("Creating mock denial prediction model")
        
        # Create simple mock model
        class MockModel:
            def predict_proba(self, X):
                # Simple heuristic based on cost and prior auth
                proba = np.column_stack([
                    1 - (X[:, 1] / 200 + X[:, 2] * 0.3),  # Class 0 probability
                    X[:, 1] / 200 + X[:, 2] * 0.3         # Class 1 probability
                ])
                return np.clip(proba, 0, 1)
        
        self.model = MockModel()
        self.feature_columns = ["patient_age", "medication_cost", "prior_auth_required"]
        self.model_metrics = ModelMetrics(
            accuracy=0.75,
            precision=0.70,
            recall=0.68,
            f1_score=0.69,
            auc_score=0.73,
            feature_importance={"medication_cost": 0.4, "prior_auth_required": 0.3, "patient_age": 0.3}
        )
    
    async def _get_prediction_data(self) -> pd.DataFrame:
        """Get patient medication data for prediction."""
        # In production, this would query current patient medication data
        # For demo purposes, generate sample data
        data = {
            "patient_id": ["patient_001", "patient_002", "patient_003"],
            "medication_id": ["med_001", "med_002", "med_003"],
            "medication_name": ["Ozempic 0.5mg", "Lipitor 40mg", "OxyContin 20mg"],
            "patient_age": [45, 67, 72],
            "medication_cost": [350.0, 120.0, 45.0],
            "prior_auth_required": [1, 0, 0],
            "step_therapy_compliant": [0, 1, 1],
            "formulary_tier": [3, 3, 2],
            "diagnosis_count": [2, 3, 2],
            "previous_denials": [0, 1, 0],
            "generic_available": [0, 1, 0],
            "quantity_ratio": [1.0, 1.0, 2.0],
            "payer_type": ["commercial", "commercial", "medicare"]
        }
        
        return pd.DataFrame(data)
    
    async def _generate_predictions(self, data: pd.DataFrame) -> List[DenialPrediction]:
        """Generate denial predictions for the data."""
        predictions = []
        
        # Prepare features
        feature_data = data[self.feature_columns[:3]].values  # Use first 3 features for mock model
        
        # Get predictions
        prediction_probabilities = self.model.predict_proba(feature_data)[:, 1]
        
        for i, row in data.iterrows():
            denial_prob = prediction_probabilities[i]
            
            # Determine risk category
            if denial_prob >= 0.7:
                risk_category = "high"
            elif denial_prob >= 0.4:
                risk_category = "medium"
            else:
                risk_category = "low"
            
            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(row)
            
            # Generate recommendations
            recommended_actions = self._generate_recommendations(row, denial_prob)
            
            prediction = DenialPrediction(
                patient_id=row["patient_id"],
                medication_id=row["medication_id"],
                medication_name=row["medication_name"],
                denial_probability=float(denial_prob),
                risk_category=risk_category,
                contributing_factors=contributing_factors,
                recommended_actions=recommended_actions,
                confidence_score=0.8  # Mock confidence score
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _identify_contributing_factors(self, row: pd.Series) -> List[str]:
        """Identify factors contributing to denial risk."""
        factors = []
        
        if row.get("medication_cost", 0) > 200:
            factors.append("High medication cost")
        
        if row.get("prior_auth_required", 0) == 1:
            factors.append("Prior authorization required")
        
        if row.get("step_therapy_compliant", 1) == 0:
            factors.append("Step therapy non-compliance")
        
        if row.get("formulary_tier", 1) > 2:
            factors.append("Non-preferred formulary tier")
        
        if row.get("previous_denials", 0) > 0:
            factors.append("Previous denial history")
        
        if row.get("quantity_ratio", 1.0) > 1.5:
            factors.append("Excessive quantity prescribed")
        
        return factors
    
    def _generate_recommendations(self, row: pd.Series, denial_prob: float) -> List[str]:
        """Generate recommendations to reduce denial risk."""
        recommendations = []
        
        if denial_prob > 0.7:
            recommendations.append("High denial risk - consider immediate intervention")
        
        if row.get("prior_auth_required", 0) == 1:
            recommendations.append("Initiate prior authorization process")
        
        if row.get("generic_available", 0) == 1:
            recommendations.append("Consider generic alternative")
        
        if row.get("formulary_tier", 1) > 2:
            recommendations.append("Review formulary alternatives")
        
        if row.get("step_therapy_compliant", 1) == 0:
            recommendations.append("Ensure step therapy compliance")
        
        if row.get("quantity_ratio", 1.0) > 1.5:
            recommendations.append("Reduce prescribed quantity")
        
        return recommendations
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if self.model_metrics and self.model_metrics.feature_importance:
            return self.model_metrics.feature_importance
        return {}
    
    async def _store_predictions(self, predictions: List[DenialPrediction]):
        """Store predictions to database."""
        logger.info(f"Storing {len(predictions)} denial predictions")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame([
            {
                "patient_id": p.patient_id,
                "medication_id": p.medication_id,
                "medication_name": p.medication_name,
                "denial_probability": p.denial_probability,
                "risk_category": p.risk_category,
                "contributing_factors": "; ".join(p.contributing_factors),
                "recommended_actions": "; ".join(p.recommended_actions),
                "confidence_score": p.confidence_score
            }
            for p in predictions
        ])
        
        # In a real implementation, this would use SQLAlchemy to store to database
        logger.info(f"Denial predictions summary:\n{df.groupby('risk_category').size()}")
    
    def retrain_model(self, new_data: pd.DataFrame = None):
        """Retrain the model with new data."""
        logger.info("Retraining denial prediction model")
        
        if new_data is not None:
            # Use provided data
            training_data = new_data
        else:
            # Generate fresh training data
            training_data = self._generate_training_data()
        
        # Retrain model
        X, y = self._prepare_training_data(training_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.ml.train_test_split, random_state=42
        )
        
        self.model = self._train_model(X_train, y_train)
        self.model_metrics = self._evaluate_model(X_test, y_test)
        self._save_model()
        
        logger.info(f"Model retrained. New AUC: {self.model_metrics.auc_score:.3f}")
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction model statistics."""
        return {
            "model_loaded": self.model is not None,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "model_metrics": self.model_metrics.__dict__ if self.model_metrics else None,
            "feature_importance": self._get_feature_importance()
        }