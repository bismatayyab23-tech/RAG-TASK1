import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class MedicalRAGSystem:
    def __init__(self, vector_store_path="medical_rag/vector_store"):
        self.vector_store_path = vector_store_path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.load_vector_store()

    def load_vector_store(self):
        try:
            possible_paths = [
                self.vector_store_path,
                "medical_rag/vector_store",
                "./medical_rag/vector_store"
            ]

            loaded = False
            for path in possible_paths:
                try:
                    index_path = f"{path}/medical_faiss.index"
                    metadata_path = f"{path}/vector_metadata.pkl"

                    if os.path.exists(index_path) and os.path.exists(metadata_path):
                        self.index = faiss.read_index(index_path)
                        with open(metadata_path, "rb") as f:
                            data = pickle.load(f)
                        self.chunks = data['chunks']
                        self.metadata = data['metadata']
                        print(f" Vector store loaded from: {path}")
                        loaded = True
                        break
                except Exception as e:
                    continue

            if not loaded:
                raise Exception("Could not load vector store from any path")

        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")

    def retrieve_similar_chunks(self, query, k=5):
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            scores, indices = self.index.search(query_embedding, k*3)

            results = []
            seen_chunks = set()

            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and idx not in seen_chunks:
                    chunk_data = {
                        'content': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(score)
                    }
                    results.append(chunk_data)
                    seen_chunks.add(idx)

                    if len(results) >= k:
                        break

            return results[:k]

        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []
