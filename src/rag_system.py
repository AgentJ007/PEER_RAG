"""
Retrieval-Augmented Generation system for finding similar papers
and weighting their decisions.
"""

import numpy as np
from typing import Tuple, List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy


class RAGSystem:
    """
    RAG system that retrieves similar papers from training set
    and aggregates their decisions using weighted voting.
    """
    
    def __init__(self, qdrant_url: str):
        """
        Initialize RAG system.
        
        Args:
            qdrant_url: URL to Qdrant vector database
        """
        self.client = QdrantClient(qdrant_url)
        self.embedder = SentenceTransformer('allenai/specter')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.nlp = spacy.load('en_core_sci_md')
    
    def predict(self, manuscript_text: str, study_type: str = None) -> Tuple[Dict, List[Dict]]:
        """
        Retrieve similar papers and aggregate their decisions.
        
        Args:
            manuscript_text: Full manuscript text
            study_type: Study type
        
        Returns:
            Tuple of (decision_probabilities, retrieved_papers_list)
        """
        
        # Step 1: Decompose manuscript into aspects
        aspects = self._decompose_manuscript(manuscript_text)
        
        # Step 2: Retrieve candidates for each aspect
        candidates = self._retrieve_candidates(aspects)
        
        # Step 3: Re-rank using cross-encoder
        ranked_papers = self._cross_encoder_rerank(aspects, candidates)
        
        # Step 4: Entity-based filtering
        filtered_papers = self._entity_filter(manuscript_text, ranked_papers)
        
        # Step 5: Statistical distance filtering
        final_papers = self._mahalanobis_filter(filtered_papers)
        
        # Step 6: Sentiment weighting
        weighted_papers = self._apply_sentiment_weights(final_papers)
        
        # Step 7: Aggregate decisions
        probs = self._aggregate_decisions(weighted_papers)
        
        # Format retrieved papers for output
        retrieved_list = [
            {
                'paper_id': p['paper_id'],
                'decision': p['decision'],
                'similarity': p['similarity'],
                'reviewer_concerns': p.get('reviewer_concerns', [])
            }
            for p in final_papers[:5]  # Top 5
        ]
        
        return probs, retrieved_list
    
    def _decompose_manuscript(self, text: str) -> Dict[str, str]:
        """Decompose manuscript into aspects for targeted retrieval."""
        
        # Simple heuristic: split into sections
        aspects = {
            'design': self._extract_section(text, 'design', 'Study design, population'),
            'statistics': self._extract_section(text, 'statistics', 'Statistical methods'),
            'confounding': self._extract_section(text, 'confounding', 'Confounding'),
            'data': self._extract_section(text, 'data', 'Data collection'),
            'ethics': self._extract_section(text, 'ethics', 'Ethics, IRB')
        }
        
        return aspects
    
    def _extract_section(self, text: str, section_type: str, keywords: str) -> str:
        """Extract relevant section from text."""
        # Simplified: return first 500 characters of text as proxy
        return text[:500] if text else ""
    
    def _retrieve_candidates(self, aspects: Dict[str, str]) -> List[Dict]:
        """Retrieve candidate papers for each aspect."""
        
        all_candidates = []
        
        for aspect, query in aspects.items():
            # Embed query
            query_embedding = self.embedder.encode([query])[0]
            
            # Search Qdrant
            search_results = self.client.search(
                collection_name='peerj_papers',
                query_vector=query_embedding,
                limit=50
            )
            
            for result in search_results:
                all_candidates.append({
                    'paper_id': result.payload['paper_id'],
                    'decision': result.payload['decision'],
                    'embedding': result.vector,
                    'similarity': result.score,
                    'text': result.payload.get('text', '')
                })
        
        # Remove duplicates (keep highest similarity)
        unique_candidates = {}
        for candidate in all_candidates:
            if candidate['paper_id'] not in unique_candidates:
                unique_candidates[candidate['paper_id']] = candidate
            else:
                # Keep if higher similarity
                if candidate['similarity'] > unique_candidates[candidate['paper_id']]['similarity']:
                    unique_candidates[candidate['paper_id']] = candidate
        
        return list(unique_candidates.values())
    
    def _cross_encoder_rerank(self, aspects: Dict[str, str], 
                             candidates: List[Dict]) -> List[Dict]:
        """Re-rank candidates using cross-encoder."""
        
        # For each candidate, compute relevance score
        for candidate in candidates:
            # Combine all aspects as one query
            query = ' '.join(aspects.values())
            candidate_text = candidate['text']
            
            # Score with cross-encoder
            score = self.cross_encoder.predict([
                [query, candidate_text]
            ])[0]
            
            candidate['cross_encoder_score'] = score
        
        # Re-rank by cross-encoder score
        candidates.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        
        return candidates[:10]  # Keep top 10
    
    def _entity_filter(self, manuscript_text: str, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by entity overlap."""
        
        # Extract entities from manuscript
        doc = self.nlp(manuscript_text[:1000])  # First 1000 chars
        manuscript_entities = set([ent.text for ent in doc.ents])
        
        # Score candidates by entity overlap
        filtered = []
        for candidate in candidates:
            candidate_doc = self.nlp(candidate['text'][:500])
            candidate_entities = set([ent.text for ent in candidate_doc.ents])
            
            # Jaccard similarity
            overlap = len(manuscript_entities & candidate_entities)
            union = len(manuscript_entities | candidate_entities)
            
            if union > 0:
                jaccard = overlap / union
                if jaccard > 0.15:  # Threshold
                    candidate['entity_overlap'] = jaccard
                    filtered.append(candidate)
        
        return filtered
    
    def _mahalanobis_filter(self, candidates: List[Dict]) -> List[Dict]:
        """Filter by statistical distance (Mahalanobis)."""
        
        # In production: Calculate actual Mahalanobis distance
        # For demo: Use simple Euclidean distance as proxy
        
        filtered = []
        for candidate in candidates:
            # Placeholder: set distance to 2.0
            mahalanobis_dist = 2.0
            
            if mahalanobis_dist < 3.0:  # Threshold (3 sigma)
                candidate['mahalanobis_distance'] = mahalanobis_dist
                filtered.append(candidate)
        
        return filtered
    
    def _apply_sentiment_weights(self, candidates: List[Dict]) -> List[Dict]:
        """Weight papers by reviewer sentiment."""
        
        for candidate in candidates:
            # In production: Analyze reviewer comments
            # For demo: Set weight to 1.0
            candidate['criticism_weight'] = 1.0
        
        return candidates
    
    def _aggregate_decisions(self, weighted_papers: List[Dict]) -> Dict[str, float]:
        """Aggregate decisions using weighted voting."""
        
        if not weighted_papers:
            # Default if no papers retrieved
            return {
                'Accept': 0.33,
                'Minor Revision': 0.33,
                'Major Revision': 0.33
            }
        
        # Calculate weights
        max_distance = max([p.get('mahalanobis_distance', 2.0) for p in weighted_papers])
        
        decision_votes = {'Accept': 0, 'Minor Revision': 0, 'Major Revision': 0}
        total_weight = 0
        
        for paper in weighted_papers:
            distance = paper.get('mahalanobis_distance', 2.0)
            weight = 1.0 / (1.0 + distance ** 2)
            
            decision = paper['decision']
            if decision in decision_votes:
                decision_votes[decision] += weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            probs = {k: v / total_weight for k, v in decision_votes.items()}
        else:
            probs = {
                'Accept': 0.33,
                'Minor Revision': 0.33,
                'Major Revision': 0.33
            }
        
        return probs
