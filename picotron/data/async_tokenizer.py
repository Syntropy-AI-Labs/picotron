"""
Asynchronous dataset tokenization pipeline using concurrent processes.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable

def _tokenize_chunk(texts: List[str], tokenizer_name: str) -> List[List[int]]:
    """Worker process task to tokenize a list of text strings."""
    # Lazy import to avoid loading heavy modules on main thread startup
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(tokenizer.encode(text))
    return tokenized_texts

class AsyncTokenizerPipeline:
    """Tokenizes text datasets in parallel using ProcessPoolExecutor."""
    def __init__(self, tokenizer_name: str, num_workers: int = 4):
        self.tokenizer_name = tokenizer_name
        self.num_workers = num_workers

    def tokenize_parallel(self, texts: List[str], chunk_size: int = 1000) -> List[List[int]]:
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        tokenized_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_tokenize_chunk, chunk, self.tokenizer_name)
                for chunk in chunks
            ]
            for future in futures:
                tokenized_results.extend(future.result())
                
        return tokenized_results
