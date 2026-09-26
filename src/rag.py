from transformers.models.efficientloftr import image_processing_efficientloftr
from transformers.models.efficientloftr import image_processing_efficientloftr
from pathlib import Path
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME, EMBEDDING_MODEL, DATA_DIR, VECTOR_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_NEW_TOKENS,
    TEMPERATURE, TOP_P
)

class IndustrialRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.metadata = None

    def load_index(self):
        vector_dir = Path(VECTOR_DIR)
        emb_file = vector_dir / "embeddings.npy"
        meta_file = vector_dir / "metadata.json"
        if not emb_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                "Vector index not found. Run: python src/build_index.py"
            )
        self.embeddings = np.load(emb_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def load_llm(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def retrieve(self, question, top_k=TOP_K):
        if self.embeddings is None:
            self.load_index()

        q = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        scores = self.embeddings @ q
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = dict(self.metadata[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    def build_prompt(self, question, contexts):
        context_text = "\n\n".join(
            f"[Source {i+1}: {c['source']} | Section: {c['section']}]\n{c['text']}"
            for i, c in enumerate(contexts)
        )

        system = """You are an industrial technical support assistant.
Answer questions using only the supplied document context.
Rules:
1. Do not invent specifications, limits, procedures, or safety requirements.
2. If the documents do not contain enough information, say:
   "The provided documents do not contain enough information to answer this."
3. Give concise, technically structured answers.
4. For safety-related questions, explicitly state relevant precautions from the context.
5. Cite supporting sources in square brackets, e.g. [Source 1].
"""

        user = f"""Document context:
{context_text}

Question:
{question}

Answer using only the context above."""
        return system, user

    def answer(self, question, top_k=TOP_K):
        contexts = self.retrieve(question, top_k)
        self.load_llm()

        system, user = self.build_prompt(question, contexts)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, contexts
