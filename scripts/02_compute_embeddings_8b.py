import os
import json
import pickle
import numpy as np
from llama_cpp import Llama
from pathlib import Path

class GGUFEmbeddingEngine:
    def __init__(self, model_path: str, use_gpu: bool = True, gpu_layers: int = -1, main_gpu: int = 0):
        self.embedding_dim = 4096
        self.device = "cpu"
        n_gpu_layers = 0
        
        # Check if llama-cpp has GPU offload support
        try:
            from llama_cpp import llama_supports_gpu_offload
            has_gpu_support = llama_supports_gpu_offload()
        except Exception:
            has_gpu_support = False
            
        if use_gpu and has_gpu_support:
            self.device = "cuda"
            n_gpu_layers = gpu_layers
            print(f"[Embedding] GPU offload available. Using gpu_layers={n_gpu_layers}")
        else:
            print("[Embedding] GPU offload unavailable or disabled; falling back to CPU.")
            
        print(f"[Embedding] Initializing Llama model from {model_path}...")
        self.model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
            verbose=False
        )
        print("[Embedding] Llama model initialized successfully.")

    def _normalize_embedding(self, emb) -> np.ndarray:
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 1:
            arr = arr.reshape(-1)

        if arr.shape[0] != self.embedding_dim:
            raise ValueError(
                f"GGUF embedding dimension mismatch: expected {self.embedding_dim}, got {arr.shape[0]}"
            )

        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
            
    def compute_embeddings(self, texts: list) -> np.ndarray:
        valid_texts = [str(t).strip() for t in texts]
        if not valid_texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        all_embs = []
        total = len(valid_texts)
        print(f"Computing embeddings for {total} items...")
        for idx, text in enumerate(valid_texts):
            emb = self.model.embed(text)
            all_embs.append(self._normalize_embedding(emb))
            if (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"  Processed {idx + 1}/{total}...")
        return np.vstack(all_embs).astype(np.float32, copy=False)

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    
    children_path = data_dir / "corpus_children.json"
    ai_path = data_dir / "corpus_ai.json"
    
    print("Loading corpora...")
    with open(children_path, "r", encoding="utf-8") as f:
        children_texts = json.load(f)
    with open(ai_path, "r", encoding="utf-8") as f:
        ai_texts = json.load(f)
        
    print(f"Loaded {len(children_texts)} children segments.")
    print(f"Loaded {len(ai_texts)} AI segments.")
    
    model_path = os.environ.get(
        "QWEN3_EMBEDDING_GGUF",
        "/home/rk/models/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Qwen3 embedding GGUF not found. Set QWEN3_EMBEDDING_GGUF to the model path."
        )
    
    # Initialize engine (offloading all layers to GPU)
    engine = GGUFEmbeddingEngine(model_path=model_path, use_gpu=True, gpu_layers=-1)
    
    print("\n--- Computing Children Embeddings ---")
    children_embs = engine.compute_embeddings(children_texts)
    
    print("\n--- Computing AI Embeddings ---")
    ai_texts_only = [item["text"] if isinstance(item, dict) else item for item in ai_texts]
    ai_embs = engine.compute_embeddings(ai_texts_only)
    
    output_pkl = data_dir / "embeddings.pkl"
    print(f"\nSaving embeddings to {output_pkl}...")
    output_data = {
        "children": {
            "texts": children_texts,
            "embeddings": children_embs
        },
        "ai": {
            "texts": ai_texts,
            "embeddings": ai_embs
        }
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(output_data, f)
        
    print("Done! Embeddings computed and saved.")

if __name__ == "__main__":
    main()
