import os
import logging
from datetime import datetime
from collections import deque
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Variables de entorno ────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY no encontrada. Revisa tu archivo .env")

# ─── Configuración ───────────────────────────────────────────────────────────
DB_DIR = "./chroma_db"
MAX_HISTORY = 10
RETRIEVER_K = 4

if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"No se encontró la base de datos Chroma en: {DB_DIR}")

# ─── Modelos ─────────────────────────────────────────────────────────────────
logger.info("Cargando modelos y base de datos...")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.5,
)

vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

logger.info("¡Alexia lista para recibir consultas!")

# ─── Prompt ──────────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """Eres Alexia, una asistente inteligente y amigable.
Usa el contexto proporcionado para responder de forma natural y precisa.
Si no encuentras la respuesta en el contexto, dilo claramente en lugar de inventar.
Fecha y hora actual: {ahora}

Contexto relevante:
{context}"""

# ─── Historial por sesión ────────────────────────────────────────────────────
sesiones: dict[str, deque] = {}

def obtener_historial(session_id: str) -> deque:
    if session_id not in sesiones:
        sesiones[session_id] = deque(maxlen=MAX_HISTORY * 2)
    return sesiones[session_id]

# ─── Lógica de chat ───────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def chat_con_alexia(mensaje_usuario: str, session_id: str) -> str:
    ahora = datetime.now().strftime("%A, %d de %B de %Y a las %H:%M:%S")
    historial = obtener_historial(session_id)

    prompt_actual = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE.replace("{ahora}", ahora)),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda _: list(historial),
        }
        | prompt_actual
        | llm
        | StrOutputParser()
    )

    respuesta = chain.invoke(mensaje_usuario)

    historial.append(HumanMessage(content=mensaje_usuario))
    historial.append(AIMessage(content=respuesta))

    logger.info(f"[{session_id}] Usuario: {mensaje_usuario[:60]}...")
    return respuesta

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Alexia Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modelos Pydantic ─────────────────────────────────────────────────────────
class MensajeRequest(BaseModel):
    mensaje: str
    session_id: str = ""

class MensajeResponse(BaseModel):
    respuesta: str
    session_id: str

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    path = "./index.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return FileResponse(path)

@app.get("/styles.css")
async def styles():
    path = "./styles.css"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="styles.css no encontrado")
    return FileResponse(path)

@app.post("/chat", response_model=MensajeResponse)
async def chat(body: MensajeRequest):
    mensaje = body.mensaje.strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")

    session_id = body.session_id or str(uuid4())

    try:
        respuesta = chat_con_alexia(mensaje, session_id)
    except Exception as e:
        logger.error(f"Error al procesar mensaje: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al procesar tu mensaje")

    return MensajeResponse(respuesta=respuesta, session_id=session_id)

@app.delete("/chat/{session_id}")
async def limpiar_sesion(session_id: str):
    if session_id in sesiones:
        del sesiones[session_id]
        return {"detail": f"Sesión {session_id} eliminada"}
    raise HTTPException(status_code=404, detail="Sesión no encontrada")

@app.get("/health")
async def health():
    return {"status": "ok", "sesiones_activas": len(sesiones)}

# ─── Arranque ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
