import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CamembertModel, CamembertTokenizer
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import warnings
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    def __len__(self): 
        return len(self.sentences)
    def __getitem__(self, idx): 
        return self.sentences[idx]

class ThematicSegmenter:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', window_size=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.model = SentenceTransformer(model_name).to(self.device)

    def _compute_embeddings(self, sentences):
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    
    def _clean_segment(self, text):
        clean_text = re.sub(r'\s+', ' ', text)  # Supprime les espaces multiples
        clean_text = re.sub(r'\s*\.\s*-', '.-', clean_text)  # Nettoie ". -"
        return clean_text.strip()
    
    
    def _segment_sentence(self, text):
        if not text:
            return []
    
        EXCEPTIONS = {
            'titles': r'M\.|Mr\.|Dr\.|Mme\.',
            'abbrev': r'cf\.|etc\.|ex\.',
            'numbers': r'\d+\.\d+',
            'acronyms': r'\b(?:[a-zA-Z]\.){2,}(?=\s+[a-z])',
            'initials': r'[A-Z]\.',
            'tel': r'.\s*(?:T[ée]l[ée]phone|T[ée]l|TEL|Tel|[Rr]ens|[Rr]enseignements?)\.?:?\s*',
            'entry': r'.\s*(?:Entr[ée]e?)\.?\s*',
            'animation': r'.\s*(?:[Aa]nimations?)\.?\s*',
            'billets': r'.\s*(?:[Bb]illets?)\.?\s*',
            'ouvert': r'.\s*(?:[Ou]uvert?|[Ii]nscriptions?)\.?\s*',
            'citations': r'(?:"[^"]+")|(?:«[^»]+»)|(?:"[^"]+")'
        }
            
        protected_text = text
        exception_tokens = {}
        for name, pattern in EXCEPTIONS.items():
            matches = list(re.finditer(pattern, protected_text))
            for m in matches:
                token = f"@{name}_{len(exception_tokens)}@"
                exception_tokens[token] = m.group(0)
                protected_text = protected_text.replace(m.group(0), token)
    
        segments = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[^"\s]|")', protected_text) if s.strip()]
    
        restored_segments = segments
        for token, original in exception_tokens.items():
            restored_segments = [self._clean_segment(s.replace(token, original)) for s in restored_segments]
    
        return restored_segments
    
    def _preprocess(self, text):
        sentences = self._segment_sentence(text)
        return [s for s in sentences if len(s.split()) > 0]
    """
    def _compute_embeddings(self, sentences, batch_size=32):
        dataset = SentenceDataset(sentences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                
                masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                embeddings.append(batch_embeddings)
        
        return np.concatenate(embeddings, axis=0)
    """
    def _compute_coherence_scores(self, similarity_matrix):
        scores = []
        n = len(similarity_matrix)
    
        for i in range(n - self.window_size):
            left_window = np.mean(similarity_matrix[i:i + self.window_size, i][similarity_matrix[i:i + self.window_size, i] < 1])
            right_window = np.mean(similarity_matrix[i, i:i + self.window_size][similarity_matrix[i, i:i + self.window_size] < 1])
            
            score = left_window * right_window
            scores.append(score)
    
        scores = np.array(scores)
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    
        return scores.tolist()
    
    def _compute_adaptive_threshold(self, scores):
        if not scores:
            return 0

        scores_clean = np.array([s for s in scores if not np.isnan(s)])
    
        if len(scores_clean) == 0:
            return 0
            
        db = DBSCAN(eps=0.3, min_samples=4).fit(np.array(scores).reshape(-1, 1))
        labels = db.labels_
        if len(set(labels)) > 1:
            return np.mean([np.mean(np.array(scores)[labels == i]) for i in set(labels) if i != -1])
        return np.percentile(scores, 25)
    
    def _refine_boundaries(self, scores, initial_boundaries):
        peaks, _ = find_peaks(-np.array(scores), distance=self.window_size, prominence=0.1)
        return sorted(set(min(peaks, key=lambda x: abs(x - b), default=b) for b in initial_boundaries))
    
    def _merge_small_segments(self, segments, embeddings):
        if not segments:
            return []
            
        segment_info = []
        current_idx = 0
        for i, segment in enumerate(segments):
            segment_info.append({
                'start': current_idx,
                'end': current_idx + len(segment),
                'original_index': i,
                'segment': segment
            })
            current_idx += len(segment)
        
        segment_info.sort(key=lambda x: x['start'])

        final_segments = []
        i = 0
        while i < len(segment_info):
            current = segment_info[i]

            if len(current['segment']) == 1:
                current_embedding = embeddings[current['start']]
                best_match = None
                best_similarity = -1

                for j in [-1, 1]:
                    adj_idx = i + j
                    if 0 <= adj_idx < len(segment_info):
                        adj_segment = segment_info[adj_idx]
                        adj_embeddings = embeddings[adj_segment['start']:adj_segment['end']]
                        avg_embedding = np.mean(adj_embeddings, axis=0)
                        
                        similarity = cosine_similarity([current_embedding], [avg_embedding])[0, 0]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = adj_idx
                
                if best_match is not None:

                    merge_target = segment_info[best_match]
                    if best_match < i: 
                        merge_target['segment'].extend(current['segment'])
                        merge_target['end'] = current['end']
                        segment_info.pop(i)
                        i = best_match  
                    else: 
                        current['segment'].extend(merge_target['segment'])
                        current['end'] = merge_target['end']
                        segment_info.pop(best_match)
                        final_segments.append(current['segment'])
                else:
                    final_segments.append(current['segment'])
            else:
                final_segments.append(current['segment'])
            i += 1
        
        return final_segments

    def segment_text(self, text):
       
        sentences = self._preprocess(text)
        if not sentences:
            return [], [], []

        embeddings = self._compute_embeddings(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        scores = self._compute_coherence_scores(similarity_matrix)
        
        threshold = self._compute_adaptive_threshold(scores)
        initial_boundaries = [i for i, score in enumerate(scores) if score < threshold]
        refined_boundaries = self._refine_boundaries(scores, initial_boundaries)
        
        segments = []
        start = 0
        
        if not refined_boundaries:
            segments = [sentences]
        else:

            for boundary in sorted(refined_boundaries):
                current_segment = sentences[start:boundary + 1]
                if current_segment:
                    segments.append(current_segment)
                start = boundary + 1
            
            last_segment = sentences[start:]
            if last_segment:
                segments.append(last_segment)
        
        if segments:
            segments = self._merge_small_segments(segments, embeddings)
        
        return segments, scores, refined_boundaries
    
    def plot_coherence_scores(self, scores, boundaries):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(scores)), scores, marker='o', linestyle='-', color='blue', alpha=0.6, label='Coherence Scores')
        threshold = self._compute_adaptive_threshold(scores)
        plt.axhline(y=threshold, color='green', linestyle='--', label='Adaptive Threshold')
        for boundary in boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, label='Boundary' if boundary == boundaries[0] else "")
        plt.title('Thematic Coherence Analysis')
        plt.xlabel('Sentence Position')
        plt.ylabel('Coherence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()