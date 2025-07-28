from typing import List
import numpy as np

class BaseEmbeddingExtractor:
    def extract_embeddings(self, sequences: List[str], batch_size: int = 1) -> np.ndarray:
        raise NotImplementedError("Must be implemented in subclass.")
