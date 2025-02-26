import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer  # Changement ici
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from tqdm import tqdm
from datetime import datetime
import logging

@dataclass
class SearchConfig:
    threshold_base: float = 0.2
    keyword_weight: float = 0.15
    semantic_weight: float = 0.65
    batch_size: int = 8
    max_sequence_length: int = 512

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]

class SemanticSearchEngine:
    def __init__(self, model_name: str, config: SearchConfig):
        """Initialise le moteur de recherche."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._segments_embeddings_cache = {}  # Cache uniquement pour les segments
        
        # Initialisation du modèle SBERT
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'initialisation du modèle: {e}")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def preprocess_text(self, text: str) -> str:

        text = re.sub(r'[«»"""]', '"', text)
        text = re.sub(r'[''‛]', "'", text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
        
        text = re.sub(r'(\d{1,3})\s+(\d{3})', r'\1\2', text)
        
        return text.strip()

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:

        dataset = SentenceDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc="Calcul des embeddings"):
            
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)

        return np.concatenate(embeddings, axis=0)

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:

        patterns = [
            (r'\b[A-Z][a-zÀ-ÿ]+(?:\s+[A-Z][a-zÀ-ÿ]+)*\b', 1.5),  # Noms propres
            (r'\b\d+(?:[\s,.]?\d+)*(?:\s*[€$%]|euros?)?\b', 1.3),  # Nombres et montants
            (r'\b(?:19|20)\d{2}\b', 1.2),  # Années
            (r'\b[A-Z]{2,}\b', 1.4),  # Acronymes
            (r'"[^"]+"', 1.3)  # Citations
        ]
        
        keywords = []
        text = self.preprocess_text(text)
        
        for pattern, weight in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                keywords.append((match.group(), weight))
        
        return keywords

    def compute_keyword_similarity(self, query: str, text: str) -> float:
        query_keywords = dict(self.extract_keywords(query))
        text_keywords = dict(self.extract_keywords(text))
        
        common_words = set(query_keywords.keys()) & set(text_keywords.keys())
        if not common_words:
            return 0.0
        
        similarity = sum(query_keywords[word] * text_keywords[word] 
                       for word in common_words)
        normalization = (np.sqrt(sum(w * w for w in query_keywords.values())) * 
                        np.sqrt(sum(w * w for w in text_keywords.values())))
        
        return similarity / normalization if normalization > 0 else 0.0
    
    def _get_cached_segments_embeddings(self, segments: List[str]) -> np.ndarray:

        if not self._segments_embeddings_cache:
            embeddings = self.compute_embeddings(segments)
            for segment, embedding in zip(segments, embeddings):
                self._segments_embeddings_cache[segment] = embedding
            return embeddings

        return np.array([self._segments_embeddings_cache[segment] for segment in segments])

    def search(self, query: str, documents: List[List[str]]) -> List[Dict[str, Any]]:

        flat_segments = []
        doc_references = []
        
        for doc_idx, document in enumerate(documents):
            for seg_idx, segment in enumerate(document):
                flat_segments.append(segment)
                doc_references.append((doc_idx, seg_idx))
        
        self.logger.info(f"Début de la recherche avec {len(flat_segments)} segments issus de {len(documents)} documents")

        query = self.preprocess_text(query)
        processed_segments = [self.preprocess_text(seg) for seg in flat_segments]

        query_embedding = self.compute_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)

        segment_embeddings = self._get_cached_segments_embeddings(processed_segments)

        query_embedding = normalize(query_embedding)
        segment_embeddings = normalize(segment_embeddings)
        
        results = []
        for idx, (segment, (doc_idx, seg_idx)) in enumerate(zip(processed_segments, doc_references)):

            semantic_sim = float(cosine_similarity(query_embedding, 
                                                segment_embeddings[idx].reshape(1, -1))[0, 0])
            keyword_sim = self.compute_keyword_similarity(query, segment)
        
            final_score = (
                self.config.semantic_weight * semantic_sim +
                self.config.keyword_weight * keyword_sim
            )
        
            if final_score > self.config.threshold_base:

                results.append({
                    "Document_Index": doc_idx,
                    "Segment_Index": seg_idx,
                    "Segment": flat_segments[idx],  # Segment original
                    "Score": final_score,
                    "Semantic_Score": semantic_sim,
                    "Keyword_Score": keyword_sim
                })
        

        results = sorted(results, key=lambda x: x["Score"], reverse=True)
        
        self.logger.info(f"Recherche terminée: {len(results)} résultats retournés")
        
        return results