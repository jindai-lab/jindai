from typing import Dict
from jindai.pipeline import PipelineStage
from jindai import storage

import numpy as np

faiss_enabled = True
try:
    import faiss
except ImportError:
    faiss_enabled = False
    faiss = None


class FaissIndexing(PipelineStage):

    def __init__(self, m=8, bits=16, d=768, nlist=200, database='') -> None:
        super().__init__()
        self.m = m
        self.bits = bits
        self.index = faiss.IndexFlatL2(d)
        self.database = database
        if storage.exists(self.database):
            self.quantizer = faiss.read_index(storage.expand_path(self.database))
        else:
            self.quantizer = faiss.IndexIVFPQ(self.index, d, nlist, m, bits)
            self.dbvectors = []
        
    def resolve(self, paragraph):
        self.dbvectors.append(np.frombuffer(paragraph.embedding, 'float32'))
    
    def summarize(self, result) -> Dict:
        self.quantizer.train(self.dbvectors)
        self.quantizer.add(self.dbvectors)
        faiss.write_index_binary(self.quantizer, self.database)
        return {'database': self.database}
    