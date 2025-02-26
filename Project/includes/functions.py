import re
import pandas as pd
from collections import Counter
import re
from termcolor import colored
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import re

from gliner import GLiNER
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

def clean_segment(text):
    clean_text = re.sub(r'\s+', ' ', text)  # Supprime les espaces multiples
    clean_text = re.sub(r'\s*\.\s*-', '.-', clean_text)  # Nettoie ". -"
    return clean_text.strip()


def segment_sentence(text):
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
        restored_segments = [clean_segment(s.replace(token, original)) for s in restored_segments]

    return restored_segments

def create_tokenizer(model_name="microsoft/mdeberta-v3-base"):
    return AutoTokenizer.from_pretrained(model_name)

def split_text_into_sliding_windows(text, tokenizer, max_tokens=512, stride=153):
    encoding = tokenizer(text.strip(), return_offsets_mapping=True, add_special_tokens=True)
    token_ids = encoding['input_ids']
    
    if len(token_ids) <= max_tokens:
        return [text.strip()]
    
    offset_mapping = encoding['offset_mapping']
    chunks = []
    start_token = 0
    
    while start_token < len(token_ids):
        end_token = min(start_token + max_tokens, len(token_ids))
        chunk_start = offset_mapping[start_token][0]
        chunk_end = offset_mapping[min(end_token - 1, len(offset_mapping) - 1)][1]
        chunk_text = text[chunk_start:chunk_end].strip()
        
        if end_token < len(token_ids):
            sentences = segment_sentence(chunk_text)
            if sentences and len(sentences) > 1:
                chunk_text = ' '.join(sentences[:-1])
                new_chunk_end = chunk_start + len(chunk_text)
                while end_token > start_token and offset_mapping[end_token-1][1] > new_chunk_end:
                    end_token -= 1
        
        if chunk_text:
            chunks.append(chunk_text)
        
        start_token += stride
        if end_token >= len(token_ids):
            break
    
    return chunks


def segment_recursively(segment, segmenter, max_words=256):
    previous_length = len(segment.split())
    
    while previous_length > max_words:
        sub_segments, _, _ = segmenter.segment_text(segment)
        new_segments = [' '.join(s) for s in sub_segments]
        new_total_length = sum(len(s.split()) for s in new_segments)

        if all(len(s.split()) <= max_words for s in new_segments) or new_total_length >= previous_length:
            return new_segments  

        previous_length = new_total_length
        segment = ' '.join(new_segments)

    return [segment]

def split_thematic_segment(corpus, segmenter):
    
    All_segments = []
    for i, article in corpus["Article"].items():
        generated_segments, coherence_scores, boundaries = segmenter.segment_text(article)
        merged_segments_list = []
        for idx, segment in enumerate(generated_segments):
            merged_segments = ' '.join(segment) 
            refined_segments = segment_recursively(merged_segments, segmenter, max_words=384)
            
            for refined_segment in refined_segments:
                merged_segments_list.append(refined_segment)
                
        All_segments.append(merged_segments_list)   
    return  All_segments


def evaluate_gliner_with_search_results(sample, search_results, concepts, model_inference):

    import re
    from difflib import SequenceMatcher
    
    def normalize_text(text):
        text = re.sub(r'[\n\t@_]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_segment_position(text, segment, threshold=0.8):
        if segment in text:
            return text.find(segment)
        
        best_match = 0
        best_pos = -1
        step = min(20, max(1, len(text) // 100)) 
        
        for i in range(0, len(text) - min(len(segment), len(text)) + 1, step):
            window_size = min(len(segment) + 50, len(text) - i)
            sub_text = text[i:i + window_size]
            ratio = SequenceMatcher(None, segment[:min(100, len(segment))], sub_text[:min(100, len(sub_text))]).ratio()
            if ratio > best_match and ratio > threshold:
                best_match = ratio
                best_pos = i
        
        return best_pos
    
    def has_similar_entity(predictions, pred_start, pred_end, pred_label, tolerance=5):
        for p in predictions:
            if (p[2] == pred_label and 
                abs(p[0] - pred_start) <= tolerance and 
                abs(p[1] - pred_end) <= tolerance):
                return True
        return False
    
    results = []
    
    segments_by_doc = {}
    for result in search_results:
        doc_idx = result["Document_Index"]
        segment = result["Segment"]
        
        if doc_idx not in segments_by_doc:
            segments_by_doc[doc_idx] = []
        
        segments_by_doc[doc_idx].append(segment)
    
    for doc_idx, (text, annotations) in enumerate(sample):
        reference_entities = annotations['entities']
        normalized_text = normalize_text(text)
        
        article_predictions = []
        
        segments = segments_by_doc.get(doc_idx, [])
        
        if not segments:
            print(f"Aucun segment trouvé pour le document {doc_idx}")
            results.append({
                'text': text,
                'reference': reference_entities,
                'predictions': []
            })
            continue
        
        found_segments = 0
        total_segments = len(segments)
        
        for segment in segments:
            if not segment or not segment.strip():
                continue
            
            normalized_segment = normalize_text(segment)
            
            segment_start = text.find(segment)

            if segment_start == -1:
                segment_start = normalized_text.find(normalized_segment)

            if segment_start == -1:
                segment_start = find_segment_position(normalized_text, normalized_segment, threshold=0.7)
            
            if segment_start == -1:
                print(f"Segment non trouvé (segment {segments.index(segment)+1}/{total_segments}): {segment[:50]}...")
                continue
            
            found_segments += 1
            
            try:
                predictions = model_inference.predict_entities(segment, concepts, threshold=0.7)
                
                for pred in predictions:
                    pred_start = segment_start + pred['start']
                    pred_end = segment_start + pred['end']

                    if pred_end > len(text):
                        pred_end = len(text)

                    if not has_similar_entity(article_predictions, pred_start, pred_end, pred['label']):
                        article_predictions.append([pred_start, pred_end, pred['label']])
            except Exception as e:
                print(f"Erreur lors du traitement du segment: {str(e)}")
                continue
        
        print(f"Document {doc_idx}: Segments trouvés: {found_segments}/{total_segments} ({found_segments/total_segments*100:.1f}%)")
        
        results.append({
            'text': text,
            'reference': reference_entities,
            'predictions': article_predictions
        })
    
    return results


def evaluate_gliner_segments(sample, text_segments, concepts, model_inference):

    import re
    from difflib import SequenceMatcher
    
    def normalize_text(text):
        """Normalise le texte pour faciliter la correspondance."""
        text = re.sub(r'[\n\t@_]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_segment_position(text, segment, threshold=0.8):
        """Trouve la position approximative d'un segment dans le texte."""
        if segment in text:  # Correspondance exacte
            return text.find(segment)
        
        # Recherche approximative avec ratio de correspondance
        best_match = 0
        best_pos = -1
        step = min(20, max(1, len(text) // 100))  # Pas adaptatif pour efficacité
        
        for i in range(0, len(text) - min(len(segment), len(text)) + 1, step):
            window_size = min(len(segment) + 50, len(text) - i)
            sub_text = text[i:i + window_size]
            ratio = SequenceMatcher(None, segment[:min(100, len(segment))], sub_text[:min(100, len(sub_text))]).ratio()
            if ratio > best_match and ratio > threshold:
                best_match = ratio
                best_pos = i
        
        return best_pos
    
    def has_similar_entity(predictions, pred_start, pred_end, pred_label, tolerance=5):
        """Vérifie si une entité similaire existe déjà."""
        for p in predictions:
            if (p[2] == pred_label and 
                abs(p[0] - pred_start) <= tolerance and 
                abs(p[1] - pred_end) <= tolerance):
                return True
        return False
    
    results = []
    
    for (text, annotations), segments in zip(sample, text_segments):
        reference_entities = annotations['entities']
        normalized_text = normalize_text(text)
        
        article_predictions = []
        found_segments = 0
        total_segments = len(segments)
        
        for segment in segments:
            if not segment or not segment.strip():
                continue
            
            normalized_segment = normalize_text(segment)
            
            # Essayer de trouver la position exacte d'abord
            segment_start = text.find(segment)
            
            # Si non trouvé, essayer avec le texte normalisé
            if segment_start == -1:
                segment_start = normalized_text.find(normalized_segment)
            
            # Si toujours non trouvé, essayer la correspondance approximative
            if segment_start == -1:
                segment_start = find_segment_position(normalized_text, normalized_segment, threshold=0.7)
            
            if segment_start == -1:
                print(f"Segment non trouvé (segment {segments.index(segment)+1}/{total_segments}): {segment[:50]}...")
                continue
            
            found_segments += 1
            
            try:
                # Prédire les entités dans le segment
                predictions = model_inference.predict_entities(segment, concepts, threshold=0.7)
                
                # Ajuster les positions et ajouter les prédictions
                for pred in predictions:
                    pred_start = segment_start + pred['start']
                    pred_end = segment_start + pred['end']
                    
                    # Vérifier que la prédiction est dans les limites du texte
                    if pred_end > len(text):
                        pred_end = len(text)
                    
                    # Éviter les doublons en vérifiant si une entité similaire existe déjà
                    if not has_similar_entity(article_predictions, pred_start, pred_end, pred['label']):
                        article_predictions.append([pred_start, pred_end, pred['label']])
            except Exception as e:
                print(f"Erreur lors du traitement du segment: {str(e)}")
                continue
        
        print(f"Segments trouvés: {found_segments}/{total_segments} ({found_segments/total_segments*100:.1f}%)")
        
        results.append({
            'text': text,
            'reference': reference_entities,
            'predictions': article_predictions
        })
    
    return results

def calculate_metrics(all_results):
    total_metrics = {
        'tp': 0, 'fp': 0, 'fn': 0,
        'by_type': {}
    }
    
    for result in all_results:
        ref_entities = result['reference']
        pred_entities = result['predictions']
        
        for pred in pred_entities:
            found_match = False
            pred_type = pred[2]
            
            if pred_type not in total_metrics['by_type']:
                total_metrics['by_type'][pred_type] = {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0}
            
            for ref in ref_entities:
                if (ref[2] == pred[2] and 
                    abs(ref[0] - pred[0]) <= 5 and  
                    abs(ref[1] - pred[1]) <= 5):    
                    total_metrics['tp'] += 1
                    total_metrics['by_type'][pred_type]['tp'] += 1
                    found_match = True
                    break
            
            if not found_match:
                total_metrics['fp'] += 1
                total_metrics['by_type'][pred_type]['fp'] += 1
        
        for ref in ref_entities:
            ref_type = ref[2]
            
            if ref_type not in total_metrics['by_type']:
                total_metrics['by_type'][ref_type] = {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0}
            
            total_metrics['by_type'][ref_type]['support'] += 1
            
            found_match = False
            for pred in pred_entities:
                if (ref[2] == pred[2] and
                    abs(ref[0] - pred[0]) <= 5 and
                    abs(ref[1] - pred[1]) <= 5):
                    found_match = True
                    break
            
            if not found_match:
                total_metrics['fn'] += 1
                total_metrics['by_type'][ref_type]['fn'] += 1
    
    return total_metrics

def print_metrics(metrics):
    print("\nMétriques par type d'entité :")
    print("=" * 80)
    print(f"{'Type':<20} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for entity_type, type_metrics in metrics['by_type'].items():
        tp = type_metrics['tp']
        fp = type_metrics['fp']
        fn = type_metrics['fn']
        support = type_metrics['support']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        print(f"{entity_type:<20} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10d}")
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    print("-" * 80)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    

    macro_precision = sum(all_precision) / len(all_precision) if all_precision else 0
    macro_recall = sum(all_recall) / len(all_recall) if all_recall else 0
    macro_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    
    print(f"{'micro avg':<20} {micro_precision:>10.3f} {micro_recall:>10.3f} {micro_f1:>10.3f}")
    print(f"{'macro avg':<20} {macro_precision:>10.3f} {macro_recall:>10.3f} {macro_f1:>10.3f}")
    
    print("\nStatistiques globales:")
    print(f"Nombre total d'entités dans les références: {sum(m['support'] for m in metrics['by_type'].values())}")
    print(f"Nombre total de prédictions: {total_tp + total_fp}")
    print(f"Vrais positifs (TP): {total_tp}")
    print(f"Faux positifs (FP): {total_fp}")
    print(f"Faux négatifs (FN): {total_fn}")
