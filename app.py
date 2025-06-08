import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from agno.agent import Agent
from agno.tools.function import Function
from agno.models.google.gemini import Gemini
from agno.memory.agent import AgentMemory
from agno.vectordb.chroma import ChromaDb
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
import logging
from typing import Dict, Optional
import time
from functools import lru_cache

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="Chatbot Turístico de Necoclí",
    description="API para el chatbot turístico de Necoclí usando Agno y Google Gemini",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@lru_cache(maxsize=1)
def get_llm():
    """Obtiene una instancia del LLM con caché para reutilización."""
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY no está configurada.")
        raise ValueError("La API Key de Google Gemini no está configurada.")
    return Gemini(
        api_key=GOOGLE_API_KEY,
        model_id="gemini-pro"
    )

# Configuración de la Base de Conocimiento
@lru_cache(maxsize=1)
def get_knowledge_base():
    """Obtiene una instancia de la base de conocimiento con caché."""
    embedding_model = SentenceTransformerEmbedder()
    kb = ChromaDb(
        collection="necocli_kb",
        embedder=embedding_model,
        path="data/kb"
    )
    if not kb.exists():
        kb.create()
        logger.info("Base de conocimiento creada exitosamente (no existía).")
    else:
        logger.info("Base de conocimiento cargada exitosamente (ya existía).")
    return kb

# Definición de Herramientas
def consultar_info_turistica(query: str) -> str:
    """Consulta la base de conocimiento para obtener información detallada sobre atracciones, alojamiento, gastronomía, actividades, historia, cultura, transporte o información práctica en Necoclí.

    Args:
        query (str): La consulta a realizar en la base de conocimiento.

    Returns:
        str: La información relevante encontrada o un mensaje indicando que no se encontró información.
    """
    try:
        kb = get_knowledge_base()
        results = kb.search(query=query, limit=3)
        
        if results:
            context = "\n".join([r.content for r in results])
            return context
        return "No se encontró información específica en nuestra base de datos para esa consulta sobre Necoclí."
    except Exception as e:
        logger.error(f"Error en consultar_info_turistica: {e}")
        return "Lo siento, hubo un error al consultar la información. Por favor, intenta de nuevo."

# Crear una instancia de la herramienta
ConsultarInfoTuristicaTool = Function.from_callable(consultar_info_turistica)

# Definición del Agente
class AgenteTurismoNecocli(Agent):
    name: str = "AgenteTurismoNecocli"
    description: str = """Eres un asistente de inteligencia artificial amigable y experto en turismo para Necoclí, Antioquia, Colombia. 
    Tu objetivo es proporcionar información precisa, útil y personalizada sobre atracciones, alojamiento, gastronomía, actividades, 
    historia, cultura y transporte en Necoclí. Siempre que sea posible, utiliza la herramienta 'consultar_info_turistica' para 
    obtener detalles de nuestra base de datos. Mantén un tono servicial y entusiasta."""
    
    def __init__(self, memory: Optional[AgentMemory] = None):
        super().__init__(
            llm=get_llm(),
            tools=[ConsultarInfoTuristicaTool],
            memory=memory or AgentMemory(num_memories=5)
        )

# Almacenamiento de memorias de usuario
user_memories: Dict[str, AgentMemory] = {}

# Rate limiting
RATE_LIMIT_WINDOW = 60  # segundos
RATE_LIMIT_MAX_REQUESTS = 30  # máximo de solicitudes por ventana
user_request_times: Dict[str, list] = {}

def check_rate_limit(user_id: str) -> bool:
    """Verifica si un usuario ha excedido el límite de solicitudes."""
    current_time = time.time()
    if user_id not in user_request_times:
        user_request_times[user_id] = []
    
    # Limpiar solicitudes antiguas
    user_request_times[user_id] = [
        t for t in user_request_times[user_id]
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    if len(user_request_times[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    user_request_times[user_id].append(current_time)
    return True

@app.post("/chat/")
async def chat_endpoint(message_data: dict, request: Request):
    """Endpoint para el chat."""
    user_id = message_data.get("user_id")
    user_message = message_data.get("message")

    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="user_id y message son requeridos")

    # Verificar rate limit
    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail="Has excedido el límite de solicitudes. Por favor, espera un momento."
        )

    try:
        # Obtener o crear memoria para el usuario
        if user_id not in user_memories:
            user_memories[user_id] = AgentMemory(num_memories=5)
            logger.info(f"Nueva sesión de memoria creada para el usuario: {user_id}")

        # Crear instancia del agente
        current_agent = AgenteTurismoNecocli(memory=user_memories[user_id])

        # Procesar mensaje
        response_generator = current_agent.run(user_message)
        agent_response = next(response_generator)["message"]

        return JSONResponse({
            "response": agent_response,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Por favor, intenta de nuevo más tarde."
        )

# Servir archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Endpoint raíz que sirve la página principal."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 