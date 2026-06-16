import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

logger = logging.getLogger("LegalMind.Processing.RAPTOR")

class RaptorProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "False").lower() == "true"
        self.encoder = SentenceTransformer(model_name, local_files_only=local_files_only)
        logger.info(f"Loaded embedding encoder: {model_name}")
        
        # Test if local vLLM is available
        self.vllm_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
        self.vllm_model = os.getenv("VLLM_MODEL_NAME", "unsloth/llama-3.1-8b-instruct-bnb-4bit")
        self.vllm_available = False
        
        if self.vllm_url and self.vllm_url.lower() != "none":
            import urllib.request
            try:
                # Quick health check using /models or /health
                with urllib.request.urlopen(f"{self.vllm_url}/models", timeout=0.5) as response:
                    if response.status == 200:
                        self.vllm_available = True
                        logger.info("✓ Local vLLM server detected and active.")
            except Exception:
                logger.info("Local vLLM server unreachable. Using centroid-based extractive summarization.")

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
        Synthesizes abstractive or high-quality extractive summaries of clustered topics.
        First attempts to contact a configured local vLLM API, and if unavailable,
        falls back to a centroid-based embedding extractive summarization using the
        pre-loaded SentenceTransformer encoder.
        """
        if not cluster_chunks:
            return ""

        # Try to use local vLLM if configured and running
        if self.vllm_available:
            prompt = (
                "You are a legal summarization assistant. Synthesize a concise, single-sentence summary of the following legal provisions:\n\n"
                + "\n".join(cluster_chunks)
                + "\n\nSummary:"
            )
            
            import urllib.request
            import json
            
            try:
                req = urllib.request.Request(
                    f"{self.vllm_url}/completions",
                    data=json.dumps({
                        "model": self.vllm_model,
                        "prompt": prompt,
                        "max_tokens": 128,
                        "temperature": 0.0
                    }).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    vllm_summary = res_data['choices'][0]['text'].strip()
                    if vllm_summary:
                        return f"SUMMARY OF PROVISIONS AND PRECEDENTS:\n{vllm_summary}"
            except Exception:
                pass

        # Fallback to Centroid-based Extractive Summarization using pre-loaded self.encoder
        try:
            # Embed all chunks in the cluster
            embeddings = self.encoder.encode(cluster_chunks)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Compute centroid embedding
            centroid = np.mean(embeddings, axis=0)
            
            # Compute cosine similarities between each chunk and the centroid
            centroid_norm = np.linalg.norm(centroid)
            similarities = []
            for i, emb in enumerate(embeddings):
                emb_norm = np.linalg.norm(emb)
                if centroid_norm > 0 and emb_norm > 0:
                    similarity = np.dot(emb, centroid) / (centroid_norm * emb_norm)
                else:
                    similarity = 0.0
                similarities.append((similarity, i))
            
            # Sort by similarity descending
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Select top 2 most representative chunks
            top_indices = [idx for _, idx in similarities[:2]]
            representative_sentences = [cluster_chunks[idx].strip() for idx in top_indices]
            
            # Form clean summary
            summary_content = " ".join(representative_sentences)
            # Ensure it is not too long but informative
            if len(summary_content) > 300:
                summary_content = summary_content[:300] + "..."
            
            return f"SUMMARY OF PROVISIONS AND PRECEDENTS:\n{summary_content}"
        except Exception as e:
            logger.warning(f"Centroid summarization failed: {e}. Falling back to basic concatenation.")
            fallback_text = " ".join([c.strip() for c in cluster_chunks[:2]])
            if len(fallback_text) > 300:
                fallback_text = fallback_text[:300] + "..."
            return f"SUMMARY OF PROVISIONS AND PRECEDENTS:\n{fallback_text}"

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
