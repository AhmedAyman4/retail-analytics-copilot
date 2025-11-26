import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import glob

class LocalRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict] = []
        self.bm25 = None
        self.tokenized_corpus = []
        self._load_and_index()

    def _load_and_index(self):
        """Loads markdown files, chunks them by paragraph/header, and builds BM25 index."""
        file_paths = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        chunk_id_counter = 0
        
        for path in file_paths:
            filename = os.path.basename(path)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by double newline (paragraphs)
            raw_chunks = content.split('\n\n')
            
            for raw in raw_chunks:
                if not raw.strip():
                    continue
                
                self.chunks.append({
                    "id": f"{filename}::chunk{chunk_id_counter}",
                    "text": raw.strip(),
                    "source": filename
                })
                
                # Simple tokenization for BM25
                self.tokenized_corpus.append(raw.lower().split())
                chunk_id_counter += 1
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Returns top-k chunks with scores."""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_n:
            # Heuristic threshold: if score is 0, don't return it
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['score'] = scores[idx]
                results.append(chunk)
                
        return results