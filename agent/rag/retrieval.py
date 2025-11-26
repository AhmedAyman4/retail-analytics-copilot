import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import glob
import logging

logger = logging.getLogger(__name__)

class LocalRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict] = []
        self.bm25 = None
        self.tokenized_corpus = []
        self._load_and_index()

    def _load_and_index(self):
        """Loads markdown files, chunks them, and builds BM25 index."""
        file_paths = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        chunk_id_counter = 0
        
        for path in file_paths:
            filename = os.path.basename(path)
            clean_name = filename.replace(".md", "").replace("_", " ")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
                continue
            
            # Smart Chunking: Split by header (#) or double newline
            # This helps keep policies together
            if "#" in content:
                raw_chunks = content.split('\n#')
                # Add back the # that was split out
                raw_chunks = [("#" + c) if i > 0 else c for i, c in enumerate(raw_chunks)]
            else:
                raw_chunks = content.split('\n\n')
            
            for raw in raw_chunks:
                if not raw.strip():
                    continue
                
                # Append the filename to the text to help retrieval find the source context
                # "From product_policy: ..."
                augmented_text = f"Source: {clean_name}\nContent: {raw.strip()}"
                
                self.chunks.append({
                    "id": f"{filename}::chunk{chunk_id_counter}",
                    "text": augmented_text,
                    "source": filename,
                    "clean_name": clean_name
                })
                
                # Tokenize
                self.tokenized_corpus.append(augmented_text.lower().split())
                chunk_id_counter += 1
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Indexed {len(self.chunks)} chunks from {len(file_paths)} files.")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Returns top-k chunks with filename boosting."""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # --- FILENAME BOOSTING ---
        # If the query mentions "policy" and the file is "product_policy", boost it.
        boosted_scores = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            
            # Base boost
            final_score = score
            
            # Keyword matching in filename
            # e.g. "product policy" in query vs "product_policy.md"
            doc_name_tokens = chunk['clean_name'].split()
            matches = sum(1 for t in doc_name_tokens if t in tokenized_query)
            
            if matches > 0:
                # Significant boost (5.0) ensures relevant filenames bubble up
                final_score += (matches * 5.0)
                
            boosted_scores.append(final_score)
        
        # Get top k indices based on BOOSTED scores
        top_n = sorted(range(len(boosted_scores)), key=lambda i: boosted_scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_n:
            if boosted_scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['score'] = boosted_scores[idx]
                results.append(chunk)
                
        return results