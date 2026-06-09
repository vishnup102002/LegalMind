import sys
import os
import re
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

# Import our custom services
from app.pipeline import LegalMindPipeline
from database.graph_store import GraphStore
from database.vector_store import VectorStore
from audio.stt_gateway import ShrutamAudioTranscriber
from audio.tts_renderer import RemedialAudioGenerator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalMind.Server")

# Initialize singletons
graph_store = GraphStore()
vector_store = VectorStore()
transcriber = ShrutamAudioTranscriber()
tts_generator = RemedialAudioGenerator()
pipeline = LegalMindPipeline(threshold=float(os.getenv("CITATION_FAITHFULNESS_THRESHOLD", 0.72)))

class IngestRequest(BaseModel):
    statute_id: str
    title: str
    text: str

# Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    status: str
    response_text: str
    faithfulness_score: float

class DocumentRequest(BaseModel):
    analysis_text: str
    recipient_name: str
    sender_name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB connections & constraints on startup
    try:
        graph_store.create_constraints()
        vector_store.init_collection()
        logger.info("✓ Databases successfully verified on server startup.")
    except Exception as e:
        logger.warning(f"Database initialization deferred on startup: {e}")
    yield
    # Close connections on shutdown
    graph_store.close()

app = FastAPI(title="LegalMind API Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Returns the operational status of database connection nodes."""
    status = {"neo4j": "disconnected", "qdrant": "disconnected", "server": "healthy"}
    try:
        # Check neo4j
        with graph_store.driver.session() as s:
            s.run("RETURN 1")
            status["neo4j"] = "connected"
    except Exception:
        pass

    try:
        # Check qdrant
        vector_store.client.get_collections()
        status["qdrant"] = "connected"
    except Exception:
        pass

    return status

@app.post("/api/legal/query", response_model=QueryResponse)
async def query_legalmind(request: QueryRequest):
    """Triggers the LangGraph state machine flow with user's legal inquiry."""
    try:
        result = pipeline.run(request.query)
        return QueryResponse(
            status=result["status"],
            response_text=result["response_text"],
            faithfulness_score=result["faithfulness_score"]
        )
    except Exception as e:
        logger.error(f"Error executing pipeline run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/transcribe")
async def transcribe_audio(file_path: str):
    """ASR endpoint to process saved audio file using Shrutam-2."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file path not found.")
    
    with open(file_path, "rb") as f:
        audio_data = f.read()
    
    transcript = transcriber.transcribe(audio_data)
    return {"transcript": transcript}

@app.post("/api/voice/webrtc-offer")
async def webrtc_offer(sdp_offer: dict):
    """
    WebRTC Signaling channel offer setup.
    In production, initializes peer connections using aiortc to stream audio.
    """
    logger.info("Received WebRTC Session description offer.")
    # Stub response matching standard WebRTC answer requirements
    return {
        "status": "connected",
        "sdp_answer": {
            "type": "answer",
            "sdp": sdp_offer.get("sdp", "")
        }
    }

@app.post("/api/documents/generate")
async def generate_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    """
    Generates a print-ready legal document (PDF) based on the IRAC analysis template.
    Uses WeasyPrint for rendering.
    """
    # Create output directory
    output_dir = "data/synthesis"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "formal_notice.pdf")
    
    # We construct a simple HTML structure containing the notice layout
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ text-align: center; font-weight: bold; font-size: 20px; text-decoration: underline; }}
            .section {{ margin-top: 15px; }}
            .signature {{ margin-top: 50px; text-align: right; }}
        </style>
    </head>
    <body>
        <div class="header">FORMAL LEGAL NOTICE</div>
        <p><strong>To:</strong> {request.recipient_name}</p>
        <p><strong>From:</strong> {request.sender_name}</p>
        <div class="section">
            <h3>SUBJECT: INCIDENT ANALYSIS AND REMEDIAL ACTION REQUIRED</h3>
            <p>{request.analysis_text.replace(chr(10), '<br>')}</p>
        </div>
        <div class="signature">
            <p>_________________________</p>
            <p>Sender Signature</p>
        </div>
    </body>
    </html>
    """
    
    
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(pdf_path)
        logger.info(f"✓ PDF Notice generated at: {pdf_path}")
        return {"status": "SUCCESS", "download_url": f"/api/documents/download?file=formal_notice.pdf"}
    except Exception as e:
        logger.warning(f"WeasyPrint PDF generator failed (falling back to txt mock): {e}")
        txt_path = pdf_path.replace(".pdf", ".txt")
        with open(txt_path, "w") as f:
            f.write(html_content)
        return {"status": "SUCCESS", "download_url": f"/api/documents/download?file=formal_notice.txt"}

@app.post("/api/documents/ingest")
async def ingest_document(request: IngestRequest):
    """
    Dynamic Ingestion API:
    Parses raw text into sections, runs RAPTOR tree-building on each section,
    generates embeddings, inserts sections/statutes to Neo4j, and upserts chunks to Qdrant.
    """
    try:
        # Initialize RaptorProcessor
        from data.processing.raptor_processor import RaptorProcessor
        raptor = RaptorProcessor()
        
        # Parse text into sections using regex
        text = request.text
        section_pattern = re.compile(
            r"SECTION\s+(\d+):\s+([^\n]+)\n(.*?)(?=SECTION\s+\d+:|$)", 
            re.DOTALL | re.IGNORECASE
        )
        
        matches = section_pattern.findall(text)
        
        # Fallback: if no sections matched, treat entire text as Section 1
        if not matches:
            matches = [("1", "General Provisions", text)]
            
        # Store Statute in Neo4j
        graph_store.add_statute(request.statute_id, request.title)
        
        total_chunks = 0
        for sec_num, sec_title, sec_body in matches:
            sec_num = sec_num.strip()
            sec_title = sec_title.strip()
            sec_body = sec_body.strip()
            
            section_id = f"{request.statute_id}_sec_{sec_num}"
            citation = f"Section {sec_num}, {request.title}"
            
            # Store Section in Neo4j
            graph_store.add_section(
                statute_id=request.statute_id,
                section_id=section_id,
                title=f"Section {sec_num}: {sec_title}",
                text=sec_body,
                citation=citation
            )
            
            # Build RAPTOR tree for this section's text (Leaf layer 0 + Summary layer 1)
            tree = raptor.build_tree(sec_body, max_layers=2)
            
            # Encode chunks and upload to Qdrant
            points = []
            for layer, chunks in tree.items():
                for chunk in chunks:
                    # Generate embedding using the RAPTOR model's encoder
                    vector = raptor.encoder.encode(chunk).tolist()
                    point_id = str(uuid.uuid4())
                    
                    points.append({
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "text": chunk,
                            "citation": f"Section {sec_num}, {request.title} (RAPTOR Layer {layer})",
                            "layer_depth": layer,
                            "section_id": section_id
                        }
                    })
            
            if points:
                vector_store.upsert_chunks(points)
                total_chunks += len(points)
                
        return {
            "status": "SUCCESS",
            "message": f"Successfully ingested statute '{request.title}' with {len(matches)} sections and {total_chunks} total RAPTOR nodes."
        }
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
