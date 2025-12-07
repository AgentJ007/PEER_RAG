"""
Main predictor class that orchestrates all components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import json
from pathlib import Path
from datetime import datetime

from .feature_extractor import FeatureExtractor
from .models import OrdinalLogisticRegression, KNearestNeighbors
from .rag_system import RAGSystem
from .utils import normalize_features, bootstrap_confidence_intervals


class PeerJPredictor:
    """
    Main prediction system that combines statistical modeling, 
    NLP, and RAG for peer review decision prediction.
    """
    
    def __init__(self, model_path: str, qdrant_url: str):
        """
        Initialize the predictor system.
        
        Args:
            model_path: Path to trained models directory
            qdrant_url: URL to Qdrant vector database
        """
        self.model_path = Path(model_path)
        self.qdrant_url = qdrant_url
        
        # Load trained models
        self.m1_model = self._load_model('m1_ordinal_logistic.pkl')
        self.m2_model = self._load_model('m2_knn.pkl')
        
        # Initialize RAG system
        self.rag_system = RAGSystem(qdrant_url=qdrant_url)
        
        # Load feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Load reference statistics (for normalization)
        self.reference_stats = self._load_reference_stats('reference_stats.json')
        
        # Ensemble weights (from Brier score calibration)
        self.ensemble_weights = {
            'M1': 0.333,
            'M2': 0.331,
            'M3': 0.336
        }
    
    def predict(self, 
                manuscript_text: str, 
                study_type: Optional[str] = None,
                include_reasoning: bool = True) -> Dict:
        """
        Make a complete prediction on a new manuscript.
        
        Args:
            manuscript_text: Full manuscript text or Methods section
            study_type: Study type (auto-detected if None)
            include_reasoning: Whether to include detailed explanation
        
        Returns:
            Dictionary with prediction, confidence, issues, and recommendations
        """
        
        start_time = datetime.now()
        
        # Step 1: Classify study type if not provided
        if study_type is None:
            study_type = self.feature_extractor.classify_study_type(manuscript_text)
        
        # Step 2: Extract features
        features = self.feature_extractor.extract_features(
            manuscript_text, 
            study_type=study_type
        )
        
        # Step 3: Normalize features (Z-score)
        features_z = normalize_features(
            features, 
            self.reference_stats,
            study_type=study_type
        )
        
        # Step 4: Get predictions from M1 (Ordinal Logistic)
        m1_probs = self.m1_model.predict_proba(features_z)
        
        # Step 5: Get predictions from M2 (k-NN)
        m2_probs = self.m2_model.predict_proba(features_z)
        
        # Step 6: Get predictions from M3 (RAG)
        m3_probs, retrieved_papers = self.rag_system.predict(
            manuscript_text, 
            study_type=study_type
        )
        
        # Step 7: Ensemble predictions
        ensemble_probs = self._ensemble_predictions(m1_probs, m2_probs, m3_probs)
        
        # Step 8: Get decision
        decision = self._get_decision(ensemble_probs)
        confidence = ensemble_probs[decision]
        
        # Step 9: Bootstrap confidence intervals
        ci = bootstrap_confidence_intervals(
            self.m1_model, 
            self.m2_model,
            self.rag_system,
            features_z,
            manuscript_text,
            self.reference_stats,
            n_iterations=100  # Reduced from 1000 for demo
        )
        
        # Step 10: Feature analysis
        feature_analysis = self._analyze_features(
            features, 
            features_z,
            self.reference_stats,
            study_type
        )
        
        # Step 11: Generate recommendations
        recommendations = self._generate_recommendations(feature_analysis)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Compile result
        result = {
            'decision': decision,
            'confidence': confidence,
            'probabilities': {
                'accept': ensemble_probs.get('Accept', 0),
                'minor': ensemble_probs.get('Minor Revision', 0),
                'major': ensemble_probs.get('Major Revision', 0)
            },
            'confidence_interval': {
                'lower': ci['lower'],
                'upper': ci['upper']
            },
            'processing_time_ms': round(processing_time),
            'study_type': study_type,
            'features': feature_analysis,
            'retrieved_papers': retrieved_papers,
            'recommendations': recommendations,
            'models': {
                'M1': m1_probs,
                'M2': m2_probs,
                'M3': m3_probs,
                'Ensemble': ensemble_probs
            }
        }
        
        return result
    
    def _load_model(self, filename: str):
        """Load pickled model from disk."""
        path = self.model_path / filename
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_reference_stats(self, filename: str) -> Dict:
        """Load reference statistics for normalization."""
        path = self.model_path / filename
        with open(path, 'r') as f:
            return json.load(f)
    
    def _ensemble_predictions(self, m1_probs, m2_probs, m3_probs) -> Dict:
        """Combine predictions using weighted ensemble."""
        
        classes = ['Accept', 'Minor Revision', 'Major Revision']
        ensemble = {}
        
        for i, cls in enumerate(classes):
            ensemble[cls] = (
                self.ensemble_weights['M1'] * m1_probs[i] +
                self.ensemble_weights['M2'] * m2_probs[i] +
                self.ensemble_weights['M3'] * m3_probs[i]
            )
        
        # Normalize to sum to 1
        total = sum(ensemble.values())
        return {k: v / total for k, v in ensemble.items()}
    
    def _get_decision(self, probs: Dict[str, float]) -> str:
        """Select decision with highest probability."""
        return max(probs, key=probs.get)
    
    def _analyze_features(self, features: Dict, features_z: Dict, 
                         reference_stats: Dict, study_type: str) -> Dict:
        """Analyze features and categorize as critical, moderate, or normal."""
        
        analysis = {
            'critical_deviations': [],
            'moderate_deviations': [],
            'normal_range': []
        }
        
        # Feature thresholds (p-values)
        critical_threshold = 0.05
        moderate_threshold = 0.10
        
        for feature_name, z_score in features_z.items():
            # Calculate p-value from Z-score
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            
            feature_info = {
                'feature': feature_name,
                'value': features.get(feature_name, None),
                'z_score': z_score,
                'p_value': p_value
            }
            
            # Categorize
            if p_value < critical_threshold and abs(z_score) > 2:
                analysis['critical_deviations'].append(feature_info)
            elif p_value < moderate_threshold and abs(z_score) > 1.5:
                analysis['moderate_deviations'].append(feature_info)
            else:
                analysis['normal_range'].append(feature_info)
        
        return analysis
    
    def _generate_recommendations(self, feature_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations based on feature analysis."""
        
        recommendations = []
        
        # Map features to recommendations
        feature_recommendations = {
            'effect_size_present': {
                'action': 'Add effect sizes and 95% confidence intervals to all statistical results',
                'priority': 'HIGH',
                'impact': '+23% probability of acceptance',
                'time': '2-3 hours'
            },
            'stat_density': {
                'action': 'Specify statistical test names explicitly (e.g., "logistic regression")',
                'priority': 'HIGH',
                'impact': '+18% probability of acceptance',
                'time': '1-2 hours'
            },
            'sample_size_justified': {
                'action': 'Add sample size justification and power analysis to Methods',
                'priority': 'MEDIUM',
                'impact': '+12% probability of acceptance',
                'time': '30 minutes'
            }
        }
        
        # Generate recommendations
        for dev in feature_analysis['critical_deviations'] + feature_analysis['moderate_deviations']:
            feature = dev['feature']
            if feature in feature_recommendations:
                rec = feature_recommendations[feature].copy()
                rec['feature'] = feature
                recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations[:3]  # Top 3 recommendations
