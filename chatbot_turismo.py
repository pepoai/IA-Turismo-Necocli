from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import requests
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Cargar configuración
with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

# Inicializar la base de conocimiento
KB_PATH = Path(CONFIG['knowledge_base']['path'])

# Configurar el cliente de Hugging Face
HF_URL = CONFIG['servers']['hf-mcp-server']['url']
HF_HEADERS = CONFIG['servers']['hf-mcp-server']['headers'].copy()
HF_HEADERS['Authorization'] = HF_HEADERS['Authorization'].replace('{{HUGGINGFACE_TOKEN}}', os.getenv('HUGGINGFACE_TOKEN', ''))

app = Flask(__name__)
CORS(app)

# Cargar información detallada
with open('experiencias_turisticas.json', 'r', encoding='utf-8') as f:
    EXPERIENCIAS = json.load(f)

# Función para consultar el modelo de IA
def query_model(prompt):
    try:
        # Construir el payload
        payload = {
            "inputs": prompt,
            "parameters": CONFIG['model']['parameters']
        }
        
        # Hacer la solicitud al servidor MCP
        response = requests.post(
            f"{HF_URL}/models/{CONFIG['model']['name']}/generate",
            headers=HF_HEADERS,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['generated_text']
        else:
            logger.error(f"Error en la consulta al modelo: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error en la consulta al modelo: {str(e)}")
        return None

# Función para buscar en la base de conocimiento
def search_knowledge_base(query):
    try:
        # Buscar en la base de datos local
        results = []
        for category in EXPERIENCIAS:
            for item in EXPERIENCIAS[category]:
                if query.lower() in item['descripcion'].lower():
                    results.append(item)
        return results
    except Exception as e:
        logger.error(f"Error en la búsqueda: {str(e)}")
        return []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').lower()
        
        # Buscar en la base de conocimiento
        kb_results = search_knowledge_base(message)
        
        # Construir contexto con los resultados de la búsqueda
        context = ""
        if kb_results:
            context = "Información relevante:\n"
            for result in kb_results[:3]:  # Tomar los 3 primeros resultados
                context += f"- {result['descripcion']}\n"
        
        # Construir el prompt para el modelo
        prompt = f"""
        Eres un asistente turístico experto en Necoclí, Antioquia, Colombia.
        
        Contexto: {context}
        
        Pregunta del usuario: {message}
        
        Responde de manera amigable y proporciona información específica sobre Necoclí.
        Si no estás seguro de la respuesta, pide más detalles al usuario.
        """
        
        # Consultar el modelo
        response = query_model(prompt)
        
        if not response:
            # Si el modelo no responde, usar respuesta por defecto
            response = "Lo siento, no puedo proporcionar información específica sobre eso. ¿Podrías reformular tu pregunta o especificar qué aspecto de Necoclí te interesa?"
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error en el chat: {str(e)}")
        return jsonify({'response': 'Lo siento, hubo un error al procesar tu solicitud. Por favor, intenta nuevamente.'}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {str(e)}")
        raise
