"""
Backend para Asistente de Ventas Consultivas
============================================

Este backend implementa una API RESTful con FastAPI para asistir a vendedores en la gestión consultiva de oportunidades comerciales. Integra inteligencia artificial (OpenAI GPT-4), búsqueda semántica (FAISS) y conexión con CRM (HubSpot) para sugerir la siguiente mejor acción basada en notas del vendedor y datos del negocio.

-------------------------------------------------------------------------------
1. Arquitectura y Flujo General
-------------------------------------------------------------------------------
- Recepción de datos del CRM (HubSpot) mediante endpoints protegidos por OAuth2.
- Procesamiento y análisis de campos relevantes (notas, etapa, valor, fechas, etc.).
- Construcción de prompt compacto y relevante para la IA, priorizando la información más útil.
- Búsqueda semántica de fragmentos relevantes del playbook usando FAISS.
- Generación de sugerencia con GPT-4, con instrucciones de brevedad y concreción.
- Respuesta estructurada para la tarjeta de HubSpot.

-------------------------------------------------------------------------------
2. Justificación de Decisiones Técnicas y Parámetros
-------------------------------------------------------------------------------
- FastAPI: Elegido por su rendimiento, asincronía nativa y facilidad de documentación automática. [FastAPI Docs](https://fastapi.tiangolo.com/)
- OpenAI GPT-4: Seleccionado por su capacidad de comprensión contextual y generación de texto de alta calidad en español. Se usa el modelo `gpt-4` por su superioridad frente a `gpt-3.5-turbo` en tareas consultivas. [OpenAI Docs](https://platform.openai.com/docs/)
- FAISS: Permite búsquedas semánticas rápidas y escalables en grandes bases de conocimiento, superando búsquedas tradicionales por palabras clave. [FAISS Paper](https://arxiv.org/abs/1702.08734) | [FAISS GitHub](https://github.com/facebookresearch/faiss)
- OAuth2 con HubSpot: Se utiliza OAuth2 para garantizar la seguridad y el acceso controlado a los datos del CRM. [HubSpot OAuth Docs](https://developers.hubspot.com/docs/api/oauth-quickstart-guide)
- Pandas y Numpy: Usados para manipulación eficiente de datos tabulares y vectores numéricos.

**Parámetros clave y su justificación:**
- `max_tokens=180`: Limita la longitud de la respuesta de la IA para evitar cortes y cumplir con los límites de HubSpot.
- `temperature=0.2`: Baja aleatoriedad para respuestas más consistentes y menos creativas, priorizando la acción concreta.
- `playbook_notes` limitado a 300 caracteres: Para asegurar que el prompt no exceda el límite de tokens y la IA tenga espacio para responder.
- Fragmentos FAISS: Se seleccionan 3 fragmentos de máximo 60 caracteres para dar contexto sin saturar el prompt.

-------------------------------------------------------------------------------
3. Definiciones y Conceptos Clave
-------------------------------------------------------------------------------
- Embeddings: Representaciones vectoriales de texto que capturan significado semántico. Permiten comparar similitud entre frases/documentos. [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- FAISS: Biblioteca de Facebook AI Research para búsqueda eficiente de similitud entre vectores de alta dimensión.
- Prompt Engineering: Técnica de diseño de instrucciones para IA, buscando maximizar la relevancia y utilidad de la respuesta. [Prompt Engineering Guide](https://www.promptingguide.ai/)
- OAuth2: Protocolo estándar para autorización segura entre aplicaciones y servicios de terceros. [OAuth2](https://oauth.net/2/)

-------------------------------------------------------------------------------
4. Alternativas Consideradas y Descartadas
-------------------------------------------------------------------------------
- Frameworks: Flask y Django REST fueron considerados, pero FastAPI ofrece mejor rendimiento y asincronía.
- Modelos de IA: Cohere y Google PaLM fueron evaluados, pero OpenAI GPT-4 tiene mejor soporte en español y mayor flexibilidad.
- Búsqueda tradicional: Se descartó por su baja precisión semántica frente a FAISS.

-------------------------------------------------------------------------------
5. Supuestos, Limitaciones y Recomendaciones
-------------------------------------------------------------------------------
- Se asume que los campos del CRM están correctamente poblados y actualizados.
- El modelo de IA puede estar limitado por la longitud del prompt y la respuesta.
- El MVP prioriza claridad y acción sobre explicaciones extensas.
- La integración con HubSpot depende de la validez de los tokens OAuth.
- Para futuras mejoras, se recomienda:
    - Incluir análisis de fechas y actividad en el prompt si se requiere mayor personalización.
    - Permitir personalización del número de pasos o longitud de la respuesta.
    - Agregar soporte para otros CRMs o canales de entrada.
    - Mejorar la gestión de errores y mensajes al usuario final.
    - Implementar tests unitarios y de integración.

-------------------------------------------------------------------------------
6. Ejemplo de Uso
-------------------------------------------------------------------------------
**Entrada (notas de playbook):**
    El cliente está interesado pero considera que el precio es alto frente a la competencia. Solicita una justificación clara del valor agregado. La decisión debe tomarse antes de fin de mes porque necesitan implementar la solución cuanto antes. El área financiera está involucrada y pide una propuesta formal.

**Respuesta esperada:**
    1. Preparar presentación de valor agregado.
    2. Enviar propuesta formal antes de fin de mes.
    3. Coordinar reunión con área financiera.

-------------------------------------------------------------------------------
7. Referencias y Lecturas Recomendadas
-------------------------------------------------------------------------------
- FastAPI: https://fastapi.tiangolo.com/
- OpenAI API: https://platform.openai.com/docs/
- FAISS: https://github.com/facebookresearch/faiss
- HubSpot API: https://developers.hubspot.com/docs/api/overview
- OAuth2: https://oauth.net/2/
- Prompt Engineering: https://www.promptingguide.ai/
"""

import os, pickle
import pandas as pd
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
import logging
import sys
import traceback
import httpx
import requests
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
import time
import datetime

def setup_logger():
    """
    Configura el sistema de logging para la aplicación.

    - Utiliza el logger 'backend_debug' con nivel INFO.
    - En producción, los logs se envían a stdout para que plataformas como Railway los capturen.
    - El formato incluye timestamp, nivel y mensaje.
    - Permite monitoreo y depuración eficiente.

    Retorna:
        logging.Logger: Instancia configurada de logger.
    
    Referencias:
        https://docs.python.org/3/library/logging.html
    """
    logger = logging.getLogger('backend_debug')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

# Inicialización del logger
logger = setup_logger()
logger.info("=== INICIO DE LA APLICACIÓN ===")

# Configuración del cliente OpenAI
try:
    load_dotenv()  # Carga variables de entorno desde .env
    
    # Inicialización del cliente OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=True)  # Habilitar verificación SSL en producción
    )
    
    if not client.api_key:
        logger.warning("[ADVERTENCIA] No se encontró la variable de entorno OPENAI_API_KEY.")
    else:
        logger.info("Cliente OpenAI configurado correctamente")
except Exception as e:
    logger.error(f"Error al configurar OpenAI: {str(e)}\n{traceback.format_exc()}")
    raise

# Carga de artefactos necesarios para el sistema
try:
    logger.info("=== INICIO CARGA DE ARTEFACTOS ===")
    
    # Carga del dataset de sentencias
    logger.info("Cargando sentencias...")
    df_sent = pd.read_csv("sentencias.csv")
    logger.debug(f"Sentencias cargadas: {len(df_sent)} registros")
    
    # Carga de embeddings pre-calculados
    logger.info("Cargando embeddings...")
    with open("embeddings_openai.pkl","rb") as f:
        df_emb = pickle.load(f)
    logger.debug(f"Embeddings cargados: {len(df_emb)} registros")
    
    # Carga del mapping de índices
    logger.info("Cargando mapping...")
    mapping = pd.read_csv("mapping.csv")
    logger.debug(f"Mapping cargado: {len(mapping)} registros")
    
    # Carga del índice FAISS para búsqueda de similitud
    logger.info("Cargando índice FAISS...")
    index = faiss.read_index("playbook_index.faiss")
    logger.debug("Índice FAISS cargado correctamente")
    
    logger.info("=== CARGA DE ARTEFACTOS COMPLETADA ===")
except Exception as e:
    logger.error(f"Error al cargar los artefactos: {str(e)}\n{traceback.format_exc()}")
    raise

def oe_embedding(text):
    """
    Genera un embedding vectorial para un texto usando el modelo de OpenAI.

    Parámetros:
        text (str): Texto para el cual se generará el embedding. 
        Se recomienda que el texto sea relevante y representativo del contexto de ventas consultivas.

    Retorna:
        np.ndarray: Vector de embedding normalizado (float32).

    Justificación:
        Los embeddings permiten comparar similitud semántica entre textos, 
        lo que es fundamental para la búsqueda de fragmentos relevantes en la base de conocimiento (FAISS).
        Se utiliza el modelo 'text-embedding-3-small' de OpenAI por su balance entre costo y precisión.
    
    Referencias:
        https://platform.openai.com/docs/guides/embeddings
    """
    try:
        logger.info(f"Obteniendo embedding para el texto: {text[:60]}...")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        logger.debug(f"Embedding generado con éxito: {len(resp.data[0].embedding)} dimensiones")
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error al obtener embedding: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error al obtener embedding: {str(e)}")

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="Asistente de Ventas API",
    description="API para sugerir acciones y responder consultas de ventas",
    version="1.0.0"
)

# Variables de entorno para OAuth
HUBSPOT_CLIENT_ID = os.getenv("HUBSPOT_CLIENT_ID")
HUBSPOT_CLIENT_SECRET = os.getenv("HUBSPOT_CLIENT_SECRET")
HUBSPOT_REDIRECT_URI = os.getenv("HUBSPOT_REDIRECT_URI", "http://localhost:8003/auth/callback")
HUBSPOT_SCOPES = "crm.objects.deals.read crm.objects.deals.write crm.objects.contacts.read"

# Almacenamiento de tokens
hubspot_tokens = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": None
}

def refresh_access_token():
    """
    Renueva el access token de HubSpot usando el refresh token.

    Utiliza el flujo OAuth2 para obtener un nuevo access token cuando el actual está por expirar.
    Es fundamental para mantener la integración segura y continua con el CRM.

    Retorna:
        bool: True si el token fue renovado exitosamente, False en caso contrario.

    Referencias:
        https://developers.hubspot.com/docs/api/oauth-quickstart-guide
    """
    global hubspot_tokens
    if not hubspot_tokens["refresh_token"]:
        return False
    token_url = "https://api.hubapi.com/oauth/v1/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": HUBSPOT_CLIENT_ID,
        "client_secret": HUBSPOT_CLIENT_SECRET,
        "refresh_token": hubspot_tokens["refresh_token"]
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(token_url, data=data, headers=headers)
    if resp.status_code == 200:
        token_data = resp.json()
        hubspot_tokens["access_token"] = token_data["access_token"]
        hubspot_tokens["refresh_token"] = token_data.get("refresh_token", hubspot_tokens["refresh_token"])
        hubspot_tokens["expires_at"] = time.time() + token_data.get("expires_in", 3600)
        return True
    return False

def get_valid_access_token():
    """
    Obtiene un access token válido para la API de HubSpot, renovándolo si es necesario.

    - Si el token está por expirar (menos de 5 minutos) o ya expiró, lo renueva automáticamente.
    - Es esencial para mantener la autenticación y evitar errores de autorización.

    Retorna:
        str | None: Access token válido o None si no se pudo obtener.
    """
    global hubspot_tokens
    if not hubspot_tokens["access_token"]:
        return None
    if time.time() + 300 >= hubspot_tokens["expires_at"]:
        if not refresh_access_token():
            return None
    return hubspot_tokens["access_token"]

@app.get("/auth/login")
def hubspot_login():
    """
    Redirige al usuario a la pantalla de autorización de HubSpot (OAuth2).

    Construye la URL de autorización con los parámetros necesarios y redirige al usuario.
    Es el primer paso del flujo OAuth2 para obtener permisos de acceso al CRM.

    Retorna:
        RedirectResponse: Redirección a la URL de autorización de HubSpot.
    """
    url = (
        f"https://app.hubspot.com/oauth/authorize?client_id={HUBSPOT_CLIENT_ID}"
        f"&scope={HUBSPOT_SCOPES.replace(' ', '%20')}"
        f"&redirect_uri={HUBSPOT_REDIRECT_URI}"
    )
    return RedirectResponse(url)

@app.get("/auth/callback")
def hubspot_callback(code: str = None, error: str = None):
    """
    Endpoint de callback para el flujo OAuth2 de HubSpot.

    Recibe el código de autorización y obtiene los tokens de acceso y refresh.
    Maneja errores de autenticación y muestra mensajes apropiados al usuario.

    Parámetros:
        code (str): Código de autorización recibido de HubSpot.
        error (str): Mensaje de error si la autenticación falla.

    Retorna:
        HTMLResponse: Mensaje de éxito o error.
    """
    global hubspot_tokens
    if error:
        return HTMLResponse(f"<h2>Error de autenticación: {error}</h2>")
    if not code:
        return HTMLResponse("<h2>No se recibió código de autorización.</h2>")
    token_url = "https://api.hubapi.com/oauth/v1/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": HUBSPOT_CLIENT_ID,
        "client_secret": HUBSPOT_CLIENT_SECRET,
        "redirect_uri": HUBSPOT_REDIRECT_URI,
        "code": code
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(token_url, data=data, headers=headers)
    if resp.status_code == 200:
        token_data = resp.json()
        hubspot_tokens["access_token"] = token_data["access_token"]
        hubspot_tokens["refresh_token"] = token_data.get("refresh_token")
        hubspot_tokens["expires_at"] = time.time() + token_data.get("expires_in", 3600)
        return HTMLResponse("<h2>¡Autenticación exitosa! Los tokens fueron recibidos y almacenados.</h2>")
    else:
        return HTMLResponse(f"<h2>Error al obtener los tokens: {resp.text}</h2>")

@app.get("/hubspot/crm-card")
async def crm_card(
    userId: str = None,
    userEmail: str = None,
    associatedObjectId: str = None,
    associatedObjectType: str = None,
    portalId: str = None,
    hs_object_id: str = None
):
    """
    Endpoint principal para la integración con la tarjeta CRM de HubSpot.

    - Recibe los parámetros del negocio desde HubSpot.
    - Obtiene los datos del negocio usando la API de HubSpot.
    - Extrae los campos relevantes y llama a la función de sugerencias.
    - Devuelve la sugerencia estructurada para mostrar en la tarjeta CRM.

    Parámetros:
        userId, userEmail, associatedObjectId, associatedObjectType, portalId, hs_object_id: Parámetros enviados por HubSpot (opcional).

    Retorna:
        dict: Estructura con la sugerencia para la tarjeta CRM.
    """
    deal_id = associatedObjectId or hs_object_id
    access_token = get_valid_access_token()
    if not access_token:
        return JSONResponse(
            status_code=401,
            content={"results": [{"title": "Error", "properties": [{"label": "Error", "dataType": "STRING", "value": "No access token. Autentícate primero en /auth/login"}]}]}
        )
    url = f"https://api.hubapi.com/crm/v3/objects/deals/{deal_id}?properties=amount,dealname,dealstage,pipeline,closedate,createdate,hs_lastmodifieddate,hubspot_owner_id,playbook_notes"
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return JSONResponse(
            status_code=404,
            content={"results": [{"title": "Error", "properties": [{"label": "Error", "dataType": "STRING", "value": f"No se pudo obtener el negocio: {resp.text}"}]}]}
        )
    deal_data = resp.json()
    deal_fields = deal_data.get("properties", {})
    # Obtener el nombre de la etapa del negocio
    dealstage_id = deal_fields.get("dealstage", "")
    stage_name = dealstage_id  # Por defecto usamos el ID
    if dealstage_id:
        stage_url = f"https://api.hubapi.com/crm/v3/pipelines/deals/default/stages"
        stage_resp = requests.get(stage_url, headers=headers)
        if stage_resp.status_code == 200:
            stages_data = stage_resp.json()
            for stage in stages_data.get("results", []):
                if stage.get("id") == dealstage_id:
                    stage_name = stage.get("label", dealstage_id)
                    logger.info(f"Nombre de la etapa obtenido: {stage_name}")
                    break
        else:
            logger.warning(f"No se pudo obtener las etapas: {stage_resp.text}")
    logger.info(f"=== DEBUG DEAL RESPONSE ===")
    logger.info(f"Status code del deal: {resp.status_code}")
    logger.info(f"Respuesta completa del deal: {deal_data}")
    logger.info(f"Propiedades del deal: {deal_fields}")
    logger.info("=== DEBUG DEAL FIELDS ===")
    logger.info(f"Todas las propiedades disponibles: {deal_fields}")
    # Campos relevantes del deal que queremos considerar
    relevant_fields = {
        "amount": deal_fields.get("amount", ""),
        "dealname": deal_fields.get("dealname", ""),
        "dealstage": deal_fields.get("dealstage", ""),
        "pipeline": deal_fields.get("pipeline", ""),
        "closedate": deal_fields.get("closedate", ""),
        "createdate": deal_fields.get("createdate", ""),
        "hs_lastmodifieddate": deal_fields.get("hs_lastmodifieddate", ""),
        "hubspot_owner_id": deal_fields.get("hubspot_owner_id", ""),
        "playbook_notes": deal_fields.get("playbook_notes", ""),  # Propiedad personalizada
        "stage_name": stage_name
    }
    logger.info(f"Playbook Notes (propiedad personalizada): '{relevant_fields['playbook_notes']}'")
    logger.info(f"Owner ID: {relevant_fields['hubspot_owner_id']}")
    logger.info(f"Nombre de la etapa: {relevant_fields['stage_name']}")
    logger.info("========================")
    try:
        requestData = {
            "playbook_notes": relevant_fields["playbook_notes"].strip(),
            "deal_fields": relevant_fields
        }
        class DummyReq:
            playbook_notes = requestData["playbook_notes"]
            deal_fields = requestData["deal_fields"]
        result = await suggestion(DummyReq())
        suggestion_text = result.get("next_best_action", "No se pudo generar sugerencia.")
    except Exception as e:
        logger.error(f"Error al generar sugerencia: {str(e)}")
        suggestion_text = f"Error: {str(e)}"
    return {
        "results": [
            {
                "objectId": deal_id,
                "title": "Siguiente Mejor Acción",
                "properties": [
                    {
                        "label": "Sugerencia",
                        "dataType": "STRING",
                        "value": suggestion_text
                    }
                ]
            }
        ]
    }

# Lógica de sugerencias (reutilizada para la tarjeta CRM)
async def suggestion(req):
    """
    Genera la siguiente mejor acción para un negocio de ventas consultivas usando IA.

    Parámetros:
        req: Objeto con los siguientes atributos:
            - playbook_notes (str): Notas del vendedor sobre el negocio.
            - deal_fields (dict): Diccionario con los campos relevantes del CRM (amount, dealstage, stage_name, etc.).

    Retorna:
        dict: {'next_best_action': str} con la sugerencia generada por la IA.
    """
    try:
        logger.info("=== INICIO DE SUGERENCIA ===")
        logger.debug(f"Request recibido: {req.playbook_notes}, {req.deal_fields}")
        
        # Análisis inteligente de los campos del CRM
        amount = req.deal_fields.get('amount', '')
        dealstage = req.deal_fields.get('dealstage', '')
        stage_name = req.deal_fields.get('stage_name', '')
        closedate = req.deal_fields.get('closedate', '')
        createdate = req.deal_fields.get('createdate', '')
        hs_lastmodifieddate = req.deal_fields.get('hs_lastmodifieddate', '')
        playbook_notes = req.playbook_notes.strip()
        if len(playbook_notes) > 300:
            playbook_notes = playbook_notes[:297] + '...'
        
        # Clasificación del tamaño del negocio
        deal_size = "No especificado"
        if amount:
            try:
                amount_value = float(amount)
                if amount_value < 2000000:
                    deal_size = "PEQUEÑO (< 2M COP)"
                elif amount_value <= 10000000:
                    deal_size = "MEDIANO (2M - 10M COP)"
                else:
                    deal_size = "GRANDE (> 10M COP)"
            except:
                deal_size = f"Valor: {amount}"
        
        # Clasificación de la etapa del negocio
        stage_name = req.deal_fields.get('stage_name', '')
        stage_analysis = ""
        if stage_name:
            stage_lower = stage_name.lower()
            if any(word in stage_lower for word in ['oportunidad', 'negociación', 'propuesta']):
                stage_analysis = "ETAPA CRÍTICA: Enfócate en manejo de objeciones y seguimiento riguroso"
            elif any(word in stage_lower for word in ['reunión', 'agendada', 'presentación']):
                stage_analysis = "ETAPA DE REUNIÓN: Prepara presentación detallada y agenda de la reunión"
            elif any(word in stage_lower for word in ['facturación', 'contrato', 'cierre']):
                stage_analysis = "ETAPA DE CIERRE: Revisa mutual action plan y prepara documentación final"
            else:
                stage_analysis = f"Etapa actual: {stage_name}"
        
        # Análisis de fechas
        date_analysis = ""
        today = datetime.datetime.now()
        
        if closedate:
            try:
                close_date = datetime.datetime.fromisoformat(closedate.replace('Z', '+00:00'))
                days_to_close = (close_date - today).days
                if days_to_close < 0:
                    date_analysis = f"FECHA DE CIERRE VENCIDA hace {abs(days_to_close)} días"
                elif days_to_close <= 7:
                    date_analysis = f"CIERRE INMINENTE en {days_to_close} días"
                elif days_to_close <= 30:
                    date_analysis = f"Cierre en {days_to_close} días - Mantén momentum"
                else:
                    date_analysis = f"Cierre en {days_to_close} días"
            except:
                date_analysis = f"Fecha de cierre: {closedate}"
        
        # Análisis de actividad reciente
        activity_analysis = ""
        if hs_lastmodifieddate:
            try:
                last_modified = datetime.datetime.fromisoformat(hs_lastmodifieddate.replace('Z', '+00:00'))
                days_since_activity = (today - last_modified).days
                if days_since_activity > 7:
                    activity_analysis = f"INACTIVIDAD CRÍTICA: {days_since_activity} días sin actividad"
                elif days_since_activity > 3:
                    activity_analysis = f"ACTIVIDAD BAJA: {days_since_activity} días sin seguimiento"
                else:
                    activity_analysis = f"Actividad reciente: {days_since_activity} días"
            except:
                activity_analysis = f"Última modificación: {hs_lastmodifieddate}"
        
        # Análisis del ciclo de vida del negocio
        lifecycle_analysis = ""
        if createdate and closedate:
            try:
                create_date = datetime.datetime.fromisoformat(createdate.replace('Z', '+00:00'))
                close_date = datetime.datetime.fromisoformat(closedate.replace('Z', '+00:00'))
                total_cycle = (close_date - create_date).days
                if total_cycle < 30:
                    lifecycle_analysis = "CICLO RÁPIDO: Negocio de ciclo corto"
                elif total_cycle <= 90:
                    lifecycle_analysis = "CICLO NORMAL: Duración estándar del negocio"
                else:
                    lifecycle_analysis = "CICLO LARGO: Negocio de ciclo extendido"
            except:
                lifecycle_analysis = "Ciclo de vida: No calculable"
        
        # Prompt ultra-directo y minimalista (sin fragmentos aún)
        prompt_base = (
            f"Notas: {playbook_notes}\n"
        )
        
        # Generar el embedding sobre el prompt base
        q_emb = oe_embedding(prompt_base)
        
        # Buscar 3 fragmentos relevantes con el embedding del prompt base
        D, I = index.search(np.array([q_emb]), 3)
        snippets = [mapping.loc[mapping.idx == i, "sentencias"].values[0][:60] + '...' if len(mapping.loc[mapping.idx == i, "sentencias"].values[0]) > 63 else mapping.loc[mapping.idx == i, "sentencias"].values[0] for i in I[0]]
        logger.info("Fragmentos seleccionados:")
        for s in snippets:
            logger.info(f"- {s}")
        
        # Reconstruir el prompt final ultra-directo
        prompt = (
            f"Notas: {playbook_notes}\n"
            f"Fragmentos: {'; '.join(snippets)}\n"
            "Próximos pasos (máx 3, frases cortas, solo acciones, máx 300 caracteres, sin justificar):"
        )
        
        logger.info(f"Prompt final DEFINITIVO:\n{prompt}\n")
        
        # Usar GPT-4 para mejor análisis
        chat = client.chat.completions.create(
            model="gpt-4",  # Modelo más potente para análisis compacto
            messages=[
                {"role": "system", "content": "Eres un asistente de ventas consultivas experto. Responde solo con acciones concretas, frases cortas y sin justificar."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=180,  # Menos tokens para forzar brevedad
            temperature=0.2  # Máxima consistencia
        )
        
        reply = chat.choices[0].message.content.strip()
        # Recorte inteligente: hasta el último punto antes de 300 caracteres
        if len(reply) > 300:
            last_dot = reply.rfind('.', 0, 300)
            if last_dot != -1:
                reply = reply[:last_dot+1]
            else:
                reply = reply[:297] + '...'
        logger.info(f"Respuesta del asistente:\n{reply}\n")
        logger.info("=== FIN DE SUGERENCIA ===")
        return {"next_best_action": reply}
        
    except Exception as e:
        logger.error(f"Error en la lógica de sugerencias: {str(e)}\n{traceback.format_exc()}")
        return {"next_best_action": f"Error: {str(e)}"}

# Para arrancar: uvicorn backend:app --reload --port 8003
