"""
rag/rag_engine.py
RAG pipeline using FAISS vector store + Sentence Transformers + OpenAI GPT-4.

Flow:
  1. Index medical documents into FAISS on first run (or load existing index)
  2. On query: embed query → retrieve top-k chunks → feed to GPT-4 with context
  3. Return grounded, cited medical explanation
"""

import os
import json
import pickle
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from knowledge_base import get_all_documents


# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH     = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
CHUNK_SIZE           = 512   # characters per chunk
CHUNK_OVERLAP        = 64
TOP_K                = 4     # retrieved chunks per query


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


# ── RAG Engine ────────────────────────────────────────────────────────────────
class MedicalRAGEngine:
    """
    Retrieval-Augmented Generation engine for medical image findings.

    - Vector store: FAISS (in-memory, persisted to disk)
    - Embeddings:   sentence-transformers/all-MiniLM-L6-v2
    - LLM:          GPT-4o-mini (via OpenAI API)
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Embedding model
        print("[RAG] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dim      = self.embedder.get_sentence_embedding_dimension()

        # Storage for chunk metadata
        self.chunks:    list[str]  = []
        self.meta:      list[dict] = []
        self.index:     Optional[faiss.IndexFlatIP] = None

        # Build or load index
        self._load_or_build_index()

    # ── Index management ──────────────────────────────────────────────────────
    def _load_or_build_index(self):
        idx_file  = FAISS_INDEX_PATH + ".faiss"
        meta_file = FAISS_INDEX_PATH + ".pkl"
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

        if os.path.exists(idx_file) and os.path.exists(meta_file):
            print("[RAG] Loading existing FAISS index...")
            self.index = faiss.read_index(idx_file)
            with open(meta_file, "rb") as f:
                saved = pickle.load(f)
            self.chunks = saved["chunks"]
            self.meta   = saved["meta"]
        else:
            print("[RAG] Building FAISS index from knowledge base...")
            self._build_index()

    def _build_index(self):
        documents = get_all_documents()

        for doc in documents:
            doc_chunks = chunk_text(doc["content"])
            for chunk in doc_chunks:
                self.chunks.append(chunk)
                self.meta.append({
                    "doc_id":   doc["id"],
                    "title":    doc["title"],
                    "category": doc["category"],
                })

        embeddings = self._embed(self.chunks)
        self.index = faiss.IndexFlatIP(self.dim)   # Inner-product (cosine after norm)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Persist
        idx_file  = FAISS_INDEX_PATH + ".faiss"
        meta_file = FAISS_INDEX_PATH + ".pkl"
        faiss.write_index(self.index, idx_file)
        with open(meta_file, "wb") as f:
            pickle.dump({"chunks": self.chunks, "meta": self.meta}, f)

        print(f"[RAG] Indexed {len(self.chunks)} chunks from {len(documents)} documents.")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using SentenceTransformer."""
        return self.embedder.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K,
                 category_filter: Optional[str] = None) -> list[dict]:
        """
        Retrieve top-k relevant chunks for a query.
        Optionally filter by disease category.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self._embed([query])
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, min(top_k * 3, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self.meta[idx]
            if category_filter and m["category"].lower() != category_filter.lower():
                if m["category"] != "General":
                    continue
            results.append({
                "chunk":    self.chunks[idx],
                "score":    float(score),
                "title":    m["title"],
                "category": m["category"],
                "doc_id":   m["doc_id"],
            })
            if len(results) >= top_k:
                break

        return results

    # ── Generation ────────────────────────────────────────────────────────────
    def generate_explanation(self,
                             predicted_class: str,
                             confidence: float,
                             cv_metrics: dict,
                             class_probs: dict,
                             custom_question: Optional[str] = None) -> dict:
        """
        RAG-augmented medical explanation.
        Returns dict with 'explanation', 'sources', 'followup_questions'.
        """
        # Build retrieval query
        query = custom_question or (
            f"{predicted_class} chest X-ray findings diagnosis treatment "
            f"radiological features management"
        )

        # Retrieve relevant context
        retrieved = self.retrieve(query, top_k=TOP_K, category_filter=predicted_class)

        if not retrieved:
            retrieved = self.retrieve(query, top_k=TOP_K)

        # Build context string
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(
                f"[Source {i}: {r['title']}]\n{r['chunk']}"
            )
        context = "\n\n".join(context_parts)

        # Build class probabilities summary
        prob_summary = ", ".join(
            f"{cls}: {p*100:.1f}%" for cls, p in
            sorted(class_probs.items(), key=lambda x: -x[1])
        )

        # Compose prompt
        system_prompt = """You are an expert radiologist AI assistant. 
You provide clear, accurate medical explanations grounded in the provided clinical references.
Always:
- Reference the specific knowledge sources provided
- Explain findings in both medical and layperson terms
- Mention severity and urgency clearly
- Suggest next clinical steps
- Note limitations of AI diagnosis
- Never replace professional medical judgment"""

        if custom_question:
            user_prompt = f"""
PATIENT SCAN ANALYSIS:
- AI Predicted Diagnosis: {predicted_class} (confidence: {confidence*100:.1f}%)
- All Class Probabilities: {prob_summary}
- Image Metrics: Mean intensity={cv_metrics.get('mean_intensity',0):.1f}, 
  Edge density={cv_metrics.get('edge_density',0):.3f}, 
  Lung area ratio={cv_metrics.get('lung_area_ratio',0):.3f}

CLINICAL KNOWLEDGE BASE:
{context}

DOCTOR'S QUESTION: {custom_question}

Please answer the question using the clinical knowledge base above as your primary reference.
Cite sources by their [Source N] label when relevant.
"""
        else:
            user_prompt = f"""
PATIENT SCAN ANALYSIS:
- AI Predicted Diagnosis: {predicted_class} (confidence: {confidence*100:.1f}%)
- All Class Probabilities: {prob_summary}
- Image Metrics: Mean intensity={cv_metrics.get('mean_intensity',0):.1f}, 
  Edge density={cv_metrics.get('edge_density',0):.3f}, 
  Lung area ratio={cv_metrics.get('lung_area_ratio',0):.3f}

CLINICAL KNOWLEDGE BASE:
{context}

Please provide:
1. **Clinical Interpretation**: Explain what this finding means medically
2. **Radiological Features**: What the AI likely detected to reach this conclusion
3. **Severity Assessment**: Is this urgent? What level of care is needed?
4. **Recommended Next Steps**: Labs, follow-up imaging, specialist referral, treatment
5. **Differential Diagnosis**: Other conditions to consider given the probability distribution
6. **Patient Education**: Simple explanation for a non-medical audience

Base your response on the clinical knowledge sources provided. Cite [Source N] references.
End with a clear disclaimer that AI findings require radiologist confirmation.
"""

        # Call GPT-4 (fallback to rule-based if no API key)
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1200,
                )
                explanation = response.choices[0].message.content
            except Exception as e:
                explanation = self._rule_based_explanation(
                    predicted_class, confidence, cv_metrics, retrieved
                )
                explanation += f"\n\n⚠️ AI narrative unavailable: {e}"
        else:
            explanation = self._rule_based_explanation(
                predicted_class, confidence, cv_metrics, retrieved
            )

        # Suggested follow-up questions
        followup_questions = self._get_followup_questions(predicted_class)

        return {
            "explanation":         explanation,
            "sources":             retrieved,
            "followup_questions":  followup_questions,
            "context_used":        context,
        }

    def _rule_based_explanation(self, predicted_class: str, confidence: float,
                                 cv_metrics: dict, retrieved: list[dict]) -> str:
        """Fallback explanation when no OpenAI API key is present."""
        source_text = "\n\n".join(
            f"**{r['title']}**\n{r['chunk'][:400]}..." for r in retrieved[:2]
        )
        return f"""## AI Analysis Result: {predicted_class}

**Confidence:** {confidence*100:.1f}%

### Retrieved Clinical Context:
{source_text}

### Image Quality Metrics:
- Mean pixel intensity: {cv_metrics.get('mean_intensity', 0):.1f}
- Edge density: {cv_metrics.get('edge_density', 0):.3f}
- Estimated lung area: {cv_metrics.get('lung_area_ratio', 0)*100:.1f}% of image

> ⚕️ *Note: Add your OPENAI_API_KEY to .env for AI-powered detailed explanations.*
> This AI analysis is for decision support only and must be confirmed by a qualified radiologist.
"""

    def _get_followup_questions(self, predicted_class: str) -> list[str]:
        questions = {
            "Normal": [
                "What are the normal variants I should be aware of?",
                "When should this patient have their next chest X-ray?",
                "Are there subtle findings that might be missed on initial review?",
            ],
            "Pneumonia": [
                "What antibiotic regimen is recommended for this type of pneumonia?",
                "When should I consider a follow-up chest X-ray?",
                "What are the criteria for hospital admission vs outpatient treatment?",
            ],
            "COVID-19": [
                "How does this compare to typical influenza pneumonia on imaging?",
                "What oxygen therapy is recommended for this severity?",
                "What are the criteria for ICU admission in COVID-19 pneumonia?",
            ],
            "Tuberculosis": [
                "What TB treatment regimen should be started?",
                "What infection control measures are required?",
                "Should we test contacts of this patient?",
            ],
            "Pleural Effusion": [
                "Should we perform thoracentesis and what tests should be sent?",
                "How do we differentiate transudative from exudative effusion?",
                "What is the most likely underlying cause based on the imaging?",
            ],
        }
        return questions.get(predicted_class, [
            "What additional imaging would you recommend?",
            "What are the treatment options?",
            "What is the prognosis for this condition?",
        ])
