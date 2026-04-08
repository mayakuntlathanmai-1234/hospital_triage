"""Patient Priority Prediction using Machine Learning."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PatientPriorityPredictor:
    """
    ML-based patient priority prediction system.
    
    Predicts priority level (HIGH, MEDIUM, LOW) based on:
    - Age
    - Severity level (0-3: LOW, MODERATE, HIGH, CRITICAL)
    - Waiting time (in minutes)
    
    Output: Priority scores for each category
    """
    
    def __init__(self, model_type: str = 'logistic', pretrained: bool = True):
        """
        Initialize the priority predictor.
        
        Args:
            model_type: 'logistic' or 'random_forest'
            pretrained: Use pretrained model if available
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.model_path = Path(__file__).parent / f'priority_model_{model_type}.pkl'
        self.scaler_path = Path(__file__).parent / 'priority_scaler.pkl'
        
        # Class mapping
        self.priority_classes = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        self.priority_to_class = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        
        if pretrained and self.model_path.exists():
            self._load_model()
        else:
            self._initialize_model()
            self._train_initial_model()
    
    def _initialize_model(self):
        """Initialize model based on type."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                solver='lbfgs'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_initial_model(self):
        """Train model with synthetic data."""
        # Generate synthetic training data
        X_train, y_train = self._generate_training_data(n_samples=500)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        logger.info(f"Initial {self.model_type} model trained with synthetic data")
    
    def _generate_training_data(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data.
        
        Features: [age, severity, waiting_time]
        Target: priority level (0=LOW, 1=MEDIUM, 2=HIGH)
        """
        np.random.seed(42)
        
        # Age: 0-100
        ages = np.random.uniform(0, 100, n_samples)
        
        # Severity: 0-3 (LOW, MODERATE, HIGH, CRITICAL)
        severities = np.random.uniform(0, 4, n_samples)
        
        # Waiting time: 0-60 minutes
        waiting_times = np.random.uniform(0, 60, n_samples)
        
        # Combine features
        X = np.column_stack([ages, severities, waiting_times])
        
        # Generate labels based on rules
        # HIGH: high severity + long wait OR critical severity
        # MEDIUM: moderate severity + some wait OR high severity + short wait
        # LOW: low severity OR short wait + moderate severity
        y = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            age, severity, wait = ages[i], severities[i], waiting_times[i]
            
            if severity >= 3:  # CRITICAL
                y[i] = 2  # HIGH priority
            elif severity >= 2 and wait >= 30:  # HIGH severity + long wait
                y[i] = 2  # HIGH priority
            elif severity >= 2:  # HIGH severity
                y[i] = 1  # MEDIUM priority
            elif severity >= 1.5 and wait >= 20:  # MODERATE + some wait
                y[i] = 1  # MEDIUM priority
            else:
                y[i] = 0  # LOW priority
        
        return X, y
    
    def predict_priority(self, age: float, severity: int, waiting_time: float) -> Dict:
        """
        Predict priority for a patient.
        
        Args:
            age: Patient age (0-100)
            severity: Severity level (0=LOW, 1=MODERATE, 2=HIGH, 3=CRITICAL)
            waiting_time: Time waiting in queue (minutes)
        
        Returns:
            Dictionary with:
            - priority: 'HIGH', 'MEDIUM', or 'LOW'
            - scores: {priority: score} for each class
            - confidence: Confidence in prediction (0-1)
            - reasoning: Explanation of prediction
        """
        if not self.is_trained:
            return self._get_rule_based_priority(age, severity, waiting_time)
        
        try:
            # Prepare input
            X = np.array([[age, severity, waiting_time]])
            X_scaled = self.scaler.transform(X)
            
            # Get prediction
            prediction = self.model.predict(X_scaled)[0]
            priority = self.priority_classes[prediction]
            
            # Get probabilities/scores
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
            else:
                # For models without predict_proba, estimate confidence
                probabilities = np.array([0.33, 0.33, 0.34])
            
            scores = {
                'LOW': float(probabilities[0]),
                'MEDIUM': float(probabilities[1]),
                'HIGH': float(probabilities[2])
            }
            
            confidence = float(probabilities[prediction])
            
            # Generate reasoning
            reasoning = self._generate_reasoning(age, severity, waiting_time, priority)
            
            return {
                'priority': priority,
                'scores': scores,
                'confidence': confidence,
                'reasoning': reasoning,
                'features': {
                    'age': age,
                    'severity': severity,
                    'waiting_time': waiting_time
                }
            }
        
        except Exception as e:
            logger.error(f"Error in priority prediction: {e}")
            return self._get_rule_based_priority(age, severity, waiting_time)
    
    def batch_predict(self, patients: List[Dict]) -> List[Dict]:
        """
        Predict priorities for multiple patients.
        
        Args:
            patients: List of dicts with 'age', 'severity', 'waiting_time'
        
        Returns:
            List of prediction results
        """
        results = []
        for patient in patients:
            result = self.predict_priority(
                patient['age'],
                patient['severity'],
                patient['waiting_time']
            )
            result['patient_id'] = patient.get('patient_id')
            results.append(result)
        return results
    
    def _get_rule_based_priority(self, age: float, severity: int, waiting_time: float) -> Dict:
        """
        Fallback rule-based priority prediction.
        
        Uses expert rules instead of ML model.
        """
        priority = 'LOW'
        scores = {'LOW': 0.33, 'MEDIUM': 0.33, 'HIGH': 0.34}
        
        # Rule-based logic
        if severity >= 3:  # CRITICAL
            priority = 'HIGH'
            scores = {'LOW': 0.05, 'MEDIUM': 0.10, 'HIGH': 0.85}
        elif severity >= 2 and waiting_time >= 30:  # HIGH + long wait
            priority = 'HIGH'
            scores = {'LOW': 0.10, 'MEDIUM': 0.20, 'HIGH': 0.70}
        elif severity >= 2:  # HIGH severity
            priority = 'MEDIUM'
            scores = {'LOW': 0.20, 'MEDIUM': 0.70, 'HIGH': 0.10}
        elif severity >= 1.5 and waiting_time >= 20:  # MODERATE + wait
            priority = 'MEDIUM'
            scores = {'LOW': 0.30, 'MEDIUM': 0.60, 'HIGH': 0.10}
        elif age < 5 or age > 85:  # Very young or very old
            priority = 'MEDIUM'
            scores = {'LOW': 0.25, 'MEDIUM': 0.65, 'HIGH': 0.10}
        else:
            priority = 'LOW'
            scores = {'LOW': 0.80, 'MEDIUM': 0.15, 'HIGH': 0.05}
        
        reasoning = self._generate_reasoning(age, severity, waiting_time, priority)
        
        return {
            'priority': priority,
            'scores': scores,
            'confidence': max(scores.values()),
            'reasoning': reasoning,
            'features': {
                'age': age,
                'severity': severity,
                'waiting_time': waiting_time
            }
        }
    
    def _generate_reasoning(self, age: float, severity: int, waiting_time: float, priority: str) -> str:
        """Generate human-readable explanation for prediction."""
        severity_names = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        severity_name = severity_names[min(int(severity), 3)]
        
        reasons = []
        
        # Age factor
        if age < 5:
            reasons.append("Very young patient requires priority")
        elif age > 85:
            reasons.append("Elderly patient requires priority")
        
        # Severity factor
        if severity >= 3:
            reasons.append("Critical condition requires immediate attention")
        elif severity >= 2:
            reasons.append("High severity patient")
        elif severity >= 1.5:
            reasons.append("Moderate severity patient")
        else:
            reasons.append("Low severity patient")
        
        # Waiting time factor
        if waiting_time >= 30:
            reasons.append(f"Been waiting {waiting_time:.0f} minutes (long wait)")
        elif waiting_time >= 15:
            reasons.append(f"Been waiting {waiting_time:.0f} minutes (moderate wait)")
        
        if not reasons:
            reasons.append("Routine case")
        
        return "; ".join(reasons)
    
    def save_model(self):
        """Save trained model to disk."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load pretrained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_model()
            self._train_initial_model()
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'classes': self.priority_classes,
            'features': ['age', 'severity', 'waiting_time'],
            'output_classes': ['LOW', 'MEDIUM', 'HIGH']
        }


# Create a global predictor instance
_predictor_instance = None


def get_predictor(model_type: str = 'logistic') -> PatientPriorityPredictor:
    """Get or create the global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PatientPriorityPredictor(model_type=model_type)
    return _predictor_instance
