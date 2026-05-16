import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Cargar variables de entorno
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Error: GOOGLE_API_KEY no encontrada. Revisa tu archivo .env")

# 2. Ruta de la base de datos
DB_DIR = "./chroma_db"
if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"No se encontró la base de datos Chroma en: {DB_DIR}")

# 3. Configurar modelos
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.5
)

# 4. Cargar Chroma existente
print("Cargando base de datos Chroma existente...")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print("¡Base de datos cargada correctamente!")

# 5. Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 6. Historial de conversación (máximo 10 mensajes = 5 intercambios)
chat_history = []

# 7. Función principal de chat

def chat_con_alexia(mensaje_usuario: str) -> str:
    ahora = datetime.now().strftime("%A, %d de %B de %Y a las %H:%M:%S")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres Alexia. Usa el contexto para responder naturalmente.
Fecha y hora actual: {ahora}

Contexto:
{{context}}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda _: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    respuesta = chain.invoke(mensaje_usuario)

    chat_history.append(HumanMessage(content=mensaje_usuario))
    chat_history.append(AIMessage(content=respuesta))
    if len(chat_history) > 10:
        chat_history.pop(0)
        chat_history.pop(0)

    return respuesta
# FastAPI
app = FastAPI()

class Mensaje(BaseModel):
    mensaje: str

@app.post("/chat")
async def chat(body: Mensaje):
    respuesta = chat_con_alexia(body.mensaje)
    return {"respuesta": respuesta}

print("¡Alexia lista para recibir consultas!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)