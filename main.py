from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load the pre-trained TechWolf/JobBERT-v2 model
model = SentenceTransformer("TechWolf/JobBERT-v2")


class Texts(BaseModel):
    text1: str
    text2: str


class TextEmbedding(BaseModel):
    text: str


@app.post("/cosine_similarity")
async def compute_cosine_similarity(texts: Texts):
    if not texts.text1 or not texts.text2:
        raise HTTPException(
            status_code=400, detail="Both text1 and text2 must be provided."
        )

    # Compute embeddings for both texts
    embeddings = model.encode([texts.text1, texts.text2], convert_to_tensor=True)

    # Compute cosine similarity between the two embeddings
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1])

    return {"cosine_similarity": cosine_sim.item()}


@app.post("/get_embedding")
async def get_embedding(text_embedding: TextEmbedding):
    if not text_embedding.text:
        raise HTTPException(status_code=400, detail="Text must be provided.")

    # Retrieve the embedding for the provided text
    embedding = model.encode(text_embedding.text, convert_to_tensor=False)

    # Convert the embedding (a numpy array) to a list for JSON serialization
    embedding_list = embedding.tolist()

    return {"embedding": embedding_list}