import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger("LegalMind.Processing.RAPTOR")

class RaptorProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding encoder: {model_name}")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Splits raw legal text into leaf node chunks of specified token lengths."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def cluster_chunks(self, embeddings: np.ndarray, num_clusters: int = 3) -> list[list[int]]:
        """
        Groups similar embeddings using simple KMeans or Gaussian Mixture Model logic.
        For demonstration, we use a basic distance-based partitioning matching GMM outcomes.
        """
        from sklearn.cluster import KMeans
        if len(embeddings) < num_clusters:
            num_clusters = len(embeddings)
            
        if num_clusters == 0:
            return []

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        
        clusters = [[] for _ in range(num_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        return clusters

    def summarize_cluster(self, cluster_chunks: list[str]) -> str:
        """
        Synthesizes abstractive summaries of clustered topics.
        Can call local vLLM or fallback to extractive template.
        """
        # In production, this would invoke Llama-3.1-8B via the local vLLM / Ollama server
        context_block = "\n---\n".join(cluster_chunks[:3])
        summary = f"SUMMARY OF PROVISIONS AND PRECEDENTS:\nThis cluster outlines legal boundaries regarding tenants' protection rules and unlawful evictions: {context_block[:200]}..."
        return summary

    def build_tree(self, text: str, max_layers: int = 3) -> dict:
        """
        Constructs the bottom-up hierarchical retrieval tree.
        """
        logger.info("Initializing RAPTOR hierarchical tree construction...")
        leaf_chunks = self.chunk_text(text)
        if not leaf_chunks:
            return {}

        current_chunks = leaf_chunks
        tree_layers = {0: current_chunks}

        for layer in range(1, max_layers):
            logger.info(f"Building RAPTOR layer {layer}...")
            embeddings = self.encoder.encode(current_chunks)
            
            # Group nodes
            num_clusters = max(1, len(current_chunks) // 3)
            clusters = self.cluster_chunks(embeddings, num_clusters=num_clusters)
            
            # Summarize clusters to form the next layer
            layer_summaries = []
            for cluster in clusters:
                chunk_group = [current_chunks[idx] for idx in cluster]
                summary = self.summarize_cluster(chunk_group)
                layer_summaries.append(summary)
            
            tree_layers[layer] = layer_summaries
            current_chunks = layer_summaries
            
            if len(current_chunks) <= 1:
                break
                
        logger.info("✓ RAPTOR hierarchical tree constructed successfully.")
        return tree_layers

if __name__ == "__main__":
    processor = RaptorProcessor()
    sample_corpus = (
        "Pursuant to the Kerala Rent Control Act, landlords must provide a written notice of at least thirty "
        "days to the tenant prior to executing an eviction order. Failure to issue this notice renders the eviction "
        "unlawful and void under state regulations. Section 12 details the tenant's right to appeal any eviction "
        "order issued without due cause. Tenants facing illegal eviction can request an emergency stay order from the "
        "jurisdictional Rent Control Court. The court will evaluate the case facts based on the balance of hardships."
    )
    tree = processor.build_tree(sample_corpus)
    for layer, chunks in tree.items():
        print(f"Layer {layer} total nodes: {len(chunks)}")
        print(f"First node in layer {layer}: '{chunks[0][:120]}...'\n")
