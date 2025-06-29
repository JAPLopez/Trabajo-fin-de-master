# TFM

## Proceso de Integración con HubSpot

### Configuración del Entorno de Desarrollador HubSpot

#### 1. Creación de la Aplicación en HubSpot Developer Portal

**Paso 1: Acceso al Portal de Desarrollador**
- Acceder a [HubSpot Developer Portal](https://developers.hubspot.com/)
- Crear cuenta de desarrollador o iniciar sesión con cuenta existente
- Navegar a "Apps" en el panel principal

**Paso 2: Creación de Nueva Aplicación**
```bash
# Información básica de la aplicación
Nombre: "Sales Playbook Assistant"
Descripción: "Asistente de IA para ventas consultivas con búsqueda semántica"
Categoría: "Sales & CRM"
Audiencia: "Private App" (uso interno)
```

**Paso 3: Configuración de Scopes y Permisos**
```json
{
  "scopes": [
    "crm.objects.deals.read",
    "crm.objects.deals.write", 
    "crm.objects.companies.read",
    "crm.objects.contacts.read",
    "crm.schemas.deals.read",
    "crm.properties.deals.read"
  ],
  "redirect_urls": [
    "http://localhost:8000/auth/callback",
    "https://tu-dominio.herokuapp.com/auth/callback"
  ]
}
```

#### 2. Configuración OAuth 2.0

**Paso 1: Obtención de Credenciales**
```bash
# Credenciales generadas por HubSpot
CLIENT_ID=your_client_id_here
CLIENT_SECRET=your_client_secret_here
REDIRECT_URI=http://localhost:8000/auth/callback
```

**Paso 2: Configuración de Variables de Entorno**
```bash
# Archivo .env
HUBSPOT_CLIENT_ID=your_client_id_here
HUBSPOT_CLIENT_SECRET=your_client_secret_here
HUBSPOT_REDIRECT_URI=http://localhost:8000/auth/callback
HUBSPOT_SCOPE=crm.objects.deals.read crm.objects.deals.write crm.objects.companies.read
```

**Paso 3: Implementación del Flujo OAuth**
```python
# Configuración en backend.py
import os
from hubspot import HubSpot
from hubspot.auth.oauth import ApiException

# Configuración OAuth
HUBSPOT_CLIENT_ID = os.getenv('HUBSPOT_CLIENT_ID')
HUBSPOT_CLIENT_SECRET = os.getenv('HUBSPOT_CLIENT_SECRET')
HUBSPOT_REDIRECT_URI = os.getenv('HUBSPOT_REDIRECT_URI')
HUBSPOT_SCOPE = ['crm.objects.deals.read', 'crm.objects.deals.write']
```

#### 3. Desarrollo de Endpoints de Autenticación

**Endpoint de Autorización**
```python
@app.get("/auth/hubspot")
async def hubspot_auth():
    """
    Inicia el flujo OAuth 2.0 con HubSpot.
    Redirige al usuario a la página de autorización de HubSpot.
    """
    auth_url = f"https://app.hubspot.com/oauth/authorize"
    params = {
        "client_id": HUBSPOT_CLIENT_ID,
        "redirect_uri": HUBSPOT_REDIRECT_URI,
        "scope": " ".join(HUBSPOT_SCOPE)
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{auth_url}?{query_string}"
    
    return RedirectResponse(url=full_url)
```

**Endpoint de Callback**
```python
@app.get("/auth/callback")
async def hubspot_callback(code: str):
    """
    Maneja el callback de OAuth y obtiene el token de acceso.
    
    Args:
        code: Código de autorización de HubSpot
    """
    try:
        # Intercambiar código por token
        hubspot = HubSpot()
        tokens = hubspot.auth.oauth.default_api.create_token(
            grant_type="authorization_code",
            code=code,
            redirect_uri=HUBSPOT_REDIRECT_URI,
            client_id=HUBSPOT_CLIENT_ID,
            client_secret=HUBSPOT_CLIENT_SECRET
        )
        
        # Almacenar tokens (en producción usar base de datos segura)
        access_token = tokens.access_token
        refresh_token = tokens.refresh_token
        
        return {"message": "Autenticación exitosa", "status": "connected"}
        
    except ApiException as e:
        raise HTTPException(status_code=400, detail=f"Error de autenticación: {e}")
```

#### 4. Configuración de Tarjeta Personalizada (Custom Card)

**Paso 1: Definición de la Tarjeta en Developer Portal**
```json
{
  "type": "crm-card",
  "data": {
    "title": "Sales Playbook Assistant",
    "fetch": {
      "targetFunction": "crm-card",
      "objectTypes": ["DEAL"]
    },
    "display": {
      "properties": ["dealname", "amount", "dealstage", "closedate"]
    }
  },
  "actions": {
    "BaseUri": "https://tu-dominio.herokuapp.com"
  }
}
```

**Paso 2: Endpoint para Servir la Tarjeta**
```python
@app.post("/hubspot/card")
async def hubspot_card(request: dict):
    """
    Endpoint que sirve el contenido de la tarjeta personalizada de HubSpot.
    
    Args:
        request: Datos del objeto CRM enviados por HubSpot
        
    Returns:
        dict: Estructura de respuesta para la tarjeta
    """
    try:
        # Extraer datos del deal
        deal_data = request.get("object", {})
        deal_id = deal_data.get("objectId")
        properties = deal_data.get("properties", {})
        
        # Procesar con IA si hay notas del playbook
        playbook_notes = properties.get("playbook_notes", "")
        
        if playbook_notes:
            # Generar sugerencia usando el sistema de IA
            suggestion = await generate_sales_suggestion(
                playbook_notes=playbook_notes,
                deal_data=properties
            )
            
            card_content = {
                "results": [{
                    "objectId": deal_id,
                    "title": "Sales Playbook Assistant",
                    "properties": [{
                        "label": "Sugerencia de IA",
                        "dataType": "STRING",
                        "value": suggestion
                    }],
                    "actions": [{
                        "type": "IFRAME",
                        "width": 890,
                        "height": 748,
                        "uri": f"https://tu-dominio.herokuapp.com/card-detail/{deal_id}",
                        "label": "Ver Detalles"
                    }]
                }]
            }
        else:
            card_content = {
                "results": [{
                    "objectId": deal_id,
                    "title": "Sales Playbook Assistant",
                    "properties": [{
                        "label": "Estado",
                        "dataType": "STRING", 
                        "value": "Agregue notas del playbook para obtener sugerencias de IA"
                    }]
                }]
            }
            
        return card_content
        
    except Exception as e:
        logger.error(f"Error en tarjeta HubSpot: {e}")
        return {
            "results": [{
                "objectId": request.get("object", {}).get("objectId", "unknown"),
                "title": "Sales Playbook Assistant",
                "properties": [{
                    "label": "Error",
                    "dataType": "STRING",
                    "value": f"Error al procesar: {str(e)}"
                }]
            }]
        }
```

#### 5. Configuración de Webhooks (Opcional)

**Paso 1: Configuración en Developer Portal**
```json
{
  "webhookUrl": "https://tu-dominio.herokuapp.com/webhooks/hubspot",
  "eventType": "deal.propertyChange",
  "propertyName": "playbook_notes",
  "active": true
}
```

**Paso 2: Endpoint para Webhooks**
```python
@app.post("/webhooks/hubspot")
async def hubspot_webhook(request: dict):
    """
    Maneja webhooks de HubSpot para actualizaciones en tiempo real.
    
    Args:
        request: Datos del webhook de HubSpot
    """
    try:
        # Verificar firma del webhook (recomendado en producción)
        # verify_hubspot_signature(request)
        
        # Procesar evento
        for event in request.get("events", []):
            if event.get("eventType") == "deal.propertyChange":
                deal_id = event.get("objectId")
                property_name = event.get("propertyName")
                
                if property_name == "playbook_notes":
                    # Procesar cambio en notas del playbook
                    logger.info(f"Playbook notes updated for deal {deal_id}")
                    
        return {"status": "processed"}
        
    except Exception as e:
        logger.error(f"Error procesando webhook: {e}")
        raise HTTPException(status_code=500, detail="Error procesando webhook")
```

#### 6. Testing y Validación

**Paso 1: Testing Local**
```bash
# Ejecutar servidor local
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

# Probar endpoints
curl -X GET "http://localhost:8000/auth/hubspot"
curl -X POST "http://localhost:8000/hubspot/card" \
  -H "Content-Type: application/json" \
  -d '{"object": {"objectId": "123", "properties": {"playbook_notes": "Test note"}}}'
```

**Paso 2: Testing con ngrok (para desarrollo)**
```bash
# Instalar ngrok
npm install -g ngrok

# Exponer puerto local
ngrok http 8000

# Actualizar URLs en HubSpot Developer Portal con URL de ngrok
```

**Paso 3: Validación en HubSpot**
- Instalar la aplicación en cuenta de prueba de HubSpot
- Crear deal de prueba con campo "playbook_notes"
- Verificar que la tarjeta aparezca correctamente
- Probar funcionalidad de IA con ejemplos de test

#### 7. Despliegue en Producción

**Paso 1: Configuración de Heroku**
```bash
# Variables de entorno en Heroku
heroku config:set HUBSPOT_CLIENT_ID=your_client_id
heroku config:set HUBSPOT_CLIENT_SECRET=your_client_secret
heroku config:set HUBSPOT_REDIRECT_URI=https://tu-app.herokuapp.com/auth/callback
```

**Paso 2: Actualización de URLs en HubSpot**
- Cambiar redirect_uri en configuración OAuth
- Actualizar webhook URLs
- Actualizar BaseUri de tarjetas personalizadas

**Paso 3: Verificación Final**
- Probar flujo OAuth completo
- Verificar funcionamiento de tarjetas
- Validar webhooks (si configurados)
- Probar con datos reales

### Consideraciones de Seguridad

#### Gestión de Tokens
```python
# Implementación segura de almacenamiento de tokens
class TokenManager:
    def __init__(self):
        self.tokens = {}  # En producción: usar Redis/Base de datos
    
    def store_token(self, user_id: str, access_token: str, refresh_token: str):
        """Almacena tokens de forma segura"""
        # Encriptar tokens antes de almacenar
        encrypted_access = encrypt_token(access_token)
        encrypted_refresh = encrypt_token(refresh_token)
        
        self.tokens[user_id] = {
            "access_token": encrypted_access,
            "refresh_token": encrypted_refresh,
            "expires_at": datetime.now() + timedelta(hours=6)
        }
    
    def get_valid_token(self, user_id: str) -> str:
        """Obtiene token válido, renovándolo si es necesario"""
        token_data = self.tokens.get(user_id)
        if not token_data:
            raise HTTPException(status_code=401, detail="No authenticated")
        
        if datetime.now() > token_data["expires_at"]:
            # Renovar token usando refresh_token
            return self.refresh_access_token(user_id)
        
        return decrypt_token(token_data["access_token"])
```

#### Validación de Webhooks
```python
import hmac
import hashlib

def verify_hubspot_signature(request_body: bytes, signature: str, secret: str) -> bool:
    """
    Verifica la firma del webhook de HubSpot para seguridad.
    
    Args:
        request_body: Cuerpo de la petición en bytes
        signature: Firma enviada por HubSpot
        secret: Secret configurado en HubSpot
    
    Returns:
        bool: True si la firma es válida
    """
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        request_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)
```

### Troubleshooting Común

#### Problemas de OAuth
- **Error 400 en callback**: Verificar redirect_uri exacta
- **Scope insuficiente**: Revisar permisos en Developer Portal
- **Token expirado**: Implementar refresh token automático

#### Problemas de Tarjetas
- **Tarjeta no aparece**: Verificar configuración de objectTypes
- **Error 500 en endpoint**: Revisar logs y estructura de respuesta
- **Datos no actualizados**: Verificar caché de HubSpot

#### Problemas de Webhooks
- **Webhook no recibido**: Verificar URL pública y certificado SSL
- **Eventos duplicados**: Implementar idempotencia
- **Timeouts**: Optimizar procesamiento asíncrono

### Métricas y Monitoreo

```python
# Implementación de métricas para monitoreo
import time
from functools import wraps

def track_hubspot_calls(func):
    """Decorator para trackear llamadas a HubSpot API"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"HubSpot call {func.__name__} successful: {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"HubSpot call {func.__name__} failed: {duration:.2f}s - {e}")
            raise
    return wrapper

@track_hubspot_calls
async def get_deal_data(deal_id: str, access_token: str):
    """Obtiene datos de deal con tracking"""
    # Implementación de llamada a API
    pass
```

Esta documentación proporciona una guía completa del proceso de integración con HubSpot, desde la configuración inicial hasta el despliegue en producción, incluyendo consideraciones de seguridad y troubleshooting.
