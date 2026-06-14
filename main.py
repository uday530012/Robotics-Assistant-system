import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import src.database

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize knowledge base
kb = src.database.RoboticsKnowledgeBase()

# Load PDF from project folder
pdf_path = os.path.join("data", "robotics_qa.pdf")
kb.ingest_pdf(pdf_path)

# Request model
class QueryModel(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(data: QueryModel):

    context = kb.retrieve_context(data.message)

    prompt = f"""
You are a helpful Robotics Assistant.

Use the provided context to answer the user's question accurately.

Context:
{context}

Question:
{data.message}
"""

    def generate_chunks():
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain"
    )