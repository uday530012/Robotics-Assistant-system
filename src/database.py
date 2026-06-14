import os
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from google import genai

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            response = self.client.models.embed_content(
                # THE FIX: Restored the exact model name that works with the new SDK!
                model="gemini-embedding-2", 
                contents=doc
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

class RoboticsKnowledgeBase:
    def __init__(self):
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(
            name="robotics_data",
            embedding_function=GeminiEmbeddingFunction()
        )

    def ingest_pdf(self, file_path: str):
        print(f"Reading and processing {file_path}...")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if len(text.strip()) == 0:
            print("❌ ERROR: The PDF is completely empty or unreadable by Python!")
            return

        # Bulletproof Chunking: Cut the text every 1000 characters
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(documents=chunks, ids=ids)
        print(f"✅ SUCCESS: Embedded {len(chunks)} elements into temporary memory!")

    def retrieve_context(self, query: str) -> str:
        results = self.collection.query(query_texts=[query], n_results=3)
        if not results['documents'] or not results['documents'][0]:
            return ""
        return "\n\n---\n\n".join(results['documents'][0])