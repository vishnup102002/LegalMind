import sys
import os
import re
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

# Import our custom services
from app.pipeline import LegalMindPipeline, LLMUnavailableError
from database.graph_store import GraphStore
from database.vector_store import VectorStore
from audio.stt_gateway import ShrutamAudioTranscriber
from audio.tts_renderer import RemedialAudioGenerator
from app.whatsapp_session import WhatsAppSessionManager

session_manager = WhatsAppSessionManager()

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalMind.Server")

# Initialize singletons
graph_store = GraphStore()
vector_store = VectorStore()
transcriber = ShrutamAudioTranscriber()
tts_generator = RemedialAudioGenerator()
pipeline = LegalMindPipeline(threshold=float(os.getenv("CITATION_FAITHFULNESS_THRESHOLD", 0.72)))
from data.processing.raptor_processor import RaptorProcessor
raptor_processor = RaptorProcessor()

class IngestRequest(BaseModel):
    statute_id: str
    title: str
    text: str
    jurisdiction: str = "Central"

# Models
class Message(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    query: str
    history: List[Message] = []
    threshold: float = None

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
        history_list = [{"role": m.role, "text": m.text} for m in request.history]
        result = pipeline.run(request.query, history=history_list, threshold=request.threshold)
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

@app.post("/api/voice/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe uploaded audio file bytes directly using Shrutam-2."""
    try:
        audio_bytes = await file.read()
        transcript = transcriber.transcribe(audio_bytes)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Error during audio file transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/speak")
async def speak_text(text: str, background_tasks: BackgroundTasks):
    """Convert text to speech and return WAV audio stream using Sooktam-2."""
    try:
        import tempfile
        from fastapi.responses import FileResponse
        os.makedirs("data/tmp", exist_ok=True)
        # Create a temp file to store output audio
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", dir="data/tmp", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Render audio to temp path
        tts_generator.text_to_indic_speech(text, temp_file_path)
        
        # Background task to clean up temp file after serving
        def cleanup():
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temp voice file: {e}")
                
        background_tasks.add_task(cleanup)
        return FileResponse(temp_file_path, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error during speech synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/documents/download")
async def download_document(file: str):
    """Securely download generated PDF or TXT legal documents."""
    from fastapi.responses import FileResponse
    # Simple directory traversal prevention
    if ".." in file or "/" in file or "\\" in file:
        raise HTTPException(status_code=400, detail="Invalid document path requested.")
    
    file_path = os.path.join("data/synthesis", file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file was not found.")
        
    return FileResponse(file_path, filename=file)

@app.post("/api/documents/ingest")
async def ingest_document(request: IngestRequest):
    """
    Dynamic Ingestion API:
    Parses raw text into sections, runs RAPTOR tree-building on each section,
    generates embeddings, inserts sections/statutes to Neo4j, and upserts chunks to Qdrant.
    """
    try:
        # Use RaptorProcessor singleton
        raptor = raptor_processor
        
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
        graph_store.add_statute(request.statute_id, request.title, request.jurisdiction)
        
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
                            "section_id": section_id,
                            "jurisdiction": request.jurisdiction.lower()
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




from fastapi import Form, Request

def format_for_whatsapp(text: str) -> str:
    """Convert Markdown to WhatsApp-friendly plain text formatting"""
    import re
    # Replace double asterisks with single asterisks
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Convert subheadings to bold headers
    text = re.sub(r'^(#+)\s+(.*?)$', r'*\2*', text, flags=re.MULTILINE)
    # Replace markdown list dashes with bullets
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
    # Clean up redundant newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def send_whatsapp_text(to: str, body: str):
    import urllib.request
    import urllib.parse
    import base64
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    
    if not account_sid or not auth_token:
        logger.warning(f"Twilio credentials missing. Suppressed send to {to}: {body[:50]}...")
        return
        
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    
    # WhatsApp message limit is 1600 characters
    chunks = [body[i:i+1500] for i in range(0, len(body), 1500)]
    for chunk in chunks:
        data = {
            "From": from_number,
            "To": to,
            "Body": chunk
        }
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        auth_str = f"{account_sid}:{auth_token}"
        auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=encoded_data,
            headers={
                "Authorization": f"Basic {auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                response.read()
        except Exception as e:
            logger.error(f"Failed to send WhatsApp text to {to}: {e}")

def send_whatsapp_document(to: str, media_url: str, caption: str):
    import urllib.request
    import urllib.parse
    import base64
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    
    if not account_sid or not auth_token:
        logger.warning(f"Twilio credentials missing. Suppressed document to {to}: {media_url}")
        return
        
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    
    data = {
        "From": from_number,
        "To": to,
        "Body": caption,
        "MediaUrl": media_url
    }
    encoded_data = urllib.parse.urlencode(data).encode("utf-8")
    auth_str = f"{account_sid}:{auth_token}"
    auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=encoded_data,
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            response.read()
    except Exception as e:
        logger.error(f"Failed to send WhatsApp document to {to}: {e}")
def send_whatsapp_audio(to: str, media_url: str, caption: str = ""):
    """Send an audio message to WhatsApp via Twilio Media API."""
    import urllib.request
    import urllib.parse
    import base64
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    
    if not account_sid or not auth_token:
        logger.warning(f"Twilio credentials missing. Suppressed audio send to {to}")
        return
        
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    
    data = {
        "From": from_number,
        "To": to,
        "MediaUrl": media_url
    }
    if caption:
        data["Body"] = caption
    
    encoded_data = urllib.parse.urlencode(data).encode("utf-8")
    auth_str = f"{account_sid}:{auth_token}"
    auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=encoded_data,
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            response.read()
        logger.info(f"✓ Sent WhatsApp audio message to {to}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp audio to {to}: {e}")

def detect_text_language(text: str) -> str:
    """Detect the primary language of text based on Unicode character analysis."""
    if any(0x0D00 <= ord(c) <= 0x0D7F for c in text):
        return "ml"  # Malayalam
    elif any(0x0900 <= ord(c) <= 0x097F for c in text):
        return "hi"  # Hindi
    elif any(0x0B80 <= ord(c) <= 0x0BFF for c in text):
        return "ta"  # Tamil
    elif any(0x0C80 <= ord(c) <= 0x0CFF for c in text):
        return "kn"  # Kannada
    elif any(0x0C00 <= ord(c) <= 0x0C7F for c in text):
        return "te"  # Telugu
    return "en"

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(""),
    From: str = Form(...),
    MediaUrl0: str = Form(None),
    NumMedia: str = Form("0")
):
    phone_number = From.replace("whatsapp:", "")
    user_message = Body.strip()
    is_voice_input = False
    detected_input_lang = "en"
    
    # 1. Voice note transcription support
    try:
        num_media_int = int(NumMedia)
    except ValueError:
        num_media_int = 0

    if num_media_int > 0 and MediaUrl0:
        logger.info(f"Received WhatsApp media message from {From}. Attempting voice note transcription...")
        import urllib.request
        import base64
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if account_sid and auth_token:
            auth_str = f"{account_sid}:{auth_token}"
            auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            req = urllib.request.Request(
                MediaUrl0,
                headers={"Authorization": f"Basic {auth_b64}"}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    audio_bytes = response.read()
                    transcription = transcriber.transcribe(audio_bytes)
                    # Detect input language from audio content
                    detected_input_lang = transcriber.detect_language(audio_bytes)
                    logger.info(f"✓ Transcribed WhatsApp voice note: '{transcription}' (lang={detected_input_lang})")
                    user_message = transcription
                    is_voice_input = True
            except Exception as e:
                logger.error(f"Failed to download/transcribe WhatsApp voice note: {e}")
                user_message = "Audio transcription failed."
        else:
            logger.warning("Twilio credentials missing. Cannot download WhatsApp media.")
            user_message = "Voice message received but Twilio credentials missing on server."

    # 2. Reset session command check
    if user_message.lower() in ["/reset", "reset", "clear"]:
        session_manager.clear(phone_number)
        reply = "നിങ്ങളുടെ ചാറ്റ് സെഷൻ റീസെറ്റ് ചെയ്തിരിക്കുന്നു.\n\nYour session has been reset. How can I help you today?"
        send_whatsapp_text(From, reply)
        return {"status": "ok"}

    # 3. Load session history
    session = session_manager.load(phone_number)
    history = session.get("history", [])

    # 4. Invoke the pipeline and manage responses safely
    try:
        result = pipeline.run(user_message, history=history)
        response_text = result["response_text"]
        
        # Format message for WhatsApp
        whatsapp_reply = format_for_whatsapp(response_text)

        # 5. Extract PDF download link if present
        download_url = None
        pdf_match = re.search(r'\[DOWNLOAD_URL:(.*?)\]', whatsapp_reply)
        if pdf_match:
            raw_url = pdf_match.group(1)
            whatsapp_reply = whatsapp_reply.replace(pdf_match.group(0), "").strip()
            
            public_url = os.getenv("PUBLIC_URL", "").strip()
            if public_url:
                download_url = f"{public_url}{raw_url}"
            else:
                download_url = f"http://localhost:8080{raw_url}"

        # 6. Send response back to user
        if download_url:
            caption = "Your formal legal notice is ready. Download it using the link or view the attached document."
            if os.getenv("PUBLIC_URL"):
                send_whatsapp_document(From, download_url, caption)
            else:
                send_whatsapp_text(From, f"{caption}\n\nNotice Link: {download_url}")
        else:
            # Always send text reply
            send_whatsapp_text(From, whatsapp_reply)
        
        # 7. Voice-to-voice: If user sent a voice note, also send back an audio reply
        if is_voice_input and response_text:
            try:
                # Detect language of the response text for TTS
                response_lang = detect_text_language(response_text)
                # Use the input language if response is mixed or English
                tts_lang = detected_input_lang if detected_input_lang != "en" else response_lang
                
                # Extract the layperson section for voice reply (most useful for illiterate users)
                voice_text = response_text
                if "LAYPERSON_ML:" in response_text and tts_lang == "ml":
                    # Extract Malayalam layperson advice
                    ml_match = re.search(r'LAYPERSON_ML:\s*(.+?)(?:\n|$)', response_text)
                    if ml_match:
                        voice_text = ml_match.group(1).strip()
                elif "LAYPERSON:" in response_text:
                    # Extract English layperson advice
                    en_match = re.search(r'LAYPERSON:\s*(.+?)(?:\n|$)', response_text)
                    if en_match:
                        voice_text = en_match.group(1).strip()
                
                # Generate audio file
                import tempfile
                os.makedirs("data/tmp", exist_ok=True)
                audio_filename = f"voice_reply_{phone_number.replace('+', '')}_{uuid.uuid4().hex[:8]}.mp3"
                audio_path = os.path.join("data/tmp", audio_filename)
                tts_generator.text_to_indic_speech(voice_text, audio_path)
                
                # Serve via public URL for Twilio to fetch
                public_url = os.getenv("PUBLIC_URL", "").strip()
                if public_url:
                    # Serve the file from a static path
                    audio_serve_url = f"{public_url}/static/voice/{audio_filename}"
                    # Copy file to static/voice/ for serving
                    voice_dir = os.path.join("app", "static", "voice")
                    os.makedirs(voice_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(audio_path, os.path.join(voice_dir, audio_filename))
                    send_whatsapp_audio(From, audio_serve_url)
                    logger.info(f"✓ Voice-to-voice reply sent to {From} in language '{tts_lang}'")
                else:
                    logger.warning("PUBLIC_URL not set. Cannot send voice reply via Twilio (requires public media URL).")
                
                # Clean up temp audio file
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Voice-to-voice reply generation failed (text reply was still sent): {e}")

        # 8. Update and save session history
        history.append({"role": "user", "text": user_message})
        history.append({"role": "assistant", "text": response_text})
        session["history"] = history
        session_manager.save(phone_number, session)
        
    except LLMUnavailableError:
        logger.error(f"LLM backends unreachable for {From}", exc_info=True)
        error_msg = (
            "ക്ഷമിക്കണം, AI സേവനം താൽക്കാലികമായി ലഭ്യമല്ല. ദയവായി 30 സെക്കൻഡ് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.\n\n"
            "Sorry, the AI service is temporarily unavailable due to high demand. "
            "Please try again in 30 seconds."
        )
        send_whatsapp_text(From, error_msg)
    except Exception as e:
        logger.error(f"Error processing pipeline run for {From}: {e}", exc_info=True)
        # Provide a specific error message based on the error type
        error_str = str(e).lower()
        if "timeout" in error_str or "timed out" in error_str:
            error_msg = (
                "ക്ഷമിക്കണം, സെർവർ പ്രതികരണം സമയപരിധി കഴിഞ്ഞു.\n\n"
                "Sorry, the server response timed out. Please send your message again."
            )
        elif "rate" in error_str or "429" in error_str or "limit" in error_str:
            error_msg = (
                "ക്ഷമിക്കണം, സേവനം ഇപ്പോൾ തിരക്കിലാണ്. ദയവായി 30 സെക്കൻഡ് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.\n\n"
                "Sorry, the service is currently busy. Please wait 30 seconds and try again."
            )
        elif "connection" in error_str or "refused" in error_str:
            error_msg = (
                "ക്ഷമിക്കണം, ഡാറ്റാബേസ് കണക്ഷൻ പ്രശ്നം.\n\n"
                "Sorry, there is a database connection issue. Our team has been notified. Please try again later."
            )
        else:
            error_msg = (
                "ക്ഷമിക്കണം, ഒരു അപ്രതീക്ഷിത പിശക് സംഭവിച്ചു.\n\n"
                "Sorry, an unexpected error occurred. Please try again or send /reset to start over."
            )
        send_whatsapp_text(From, error_msg)

    return {"status": "ok"}

# Serve static files and mount index.html at root "/"
os.makedirs("app/static", exist_ok=True)
os.makedirs("app/static/voice", exist_ok=True)  # Voice reply audio files served here
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse("app/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
