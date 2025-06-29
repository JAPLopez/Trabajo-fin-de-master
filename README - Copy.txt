# TFM

# Backend para Asistente de Ventas Consultivas

Este backend implementa una API RESTful con FastAPI para asistir a vendedores en la gestión consultiva de oportunidades comerciales. Integra inteligencia artificial (OpenAI GPT-4), búsqueda semántica (FAISS) y conexión con CRM (HubSpot) para sugerir la siguiente mejor acción basada en notas del vendedor y datos del negocio.

-------------------------------------------------------------------------------
## 1. Arquitectura y Flujo General
-------------------------------------------------------------------------------
- Recepción de datos del CRM (HubSpot) mediante endpoints protegidos por OAuth2.
- Procesamiento y análisis de campos relevantes (notas, etapa, valor, fechas, etc.).
- Construcción de prompt compacto y relevante para la IA, priorizando la información más útil.
- Búsqueda semántica de fragmentos relevantes del playbook usando FAISS.
- Generación de sugerencia con GPT-4, con instrucciones de brevedad y concreción.
- Respuesta estructurada para la tarjeta de HubSpot.

-------------------------------------------------------------------------------
## 2. Justificación de Decisiones Técnicas y Parámetros
-------------------------------------------------------------------------------
- **FastAPI:** Elegido por su rendimiento, asincronía nativa y facilidad de documentación automática. [FastAPI Docs](https://fastapi.tiangolo.com/)
- **OpenAI GPT-4:** Seleccionado por su capacidad de comprensión contextual y generación de texto de alta calidad en español. Se usa el modelo `gpt-4` por su superioridad frente a `gpt-3.5-turbo` en tareas consultivas. [OpenAI Docs](https://platform.openai.com/docs/)
- **FAISS:** Permite búsquedas semánticas rápidas y escalables en grandes bases de conocimiento, superando búsquedas tradicionales por palabras clave. [FAISS Paper](https://arxiv.org/abs/1702.08734) | [FAISS GitHub](https://github.com/facebookresearch/faiss)
- **OAuth2 con HubSpot:** Se utiliza OAuth2 para garantizar la seguridad y el acceso controlado a los datos del CRM. [HubSpot OAuth Docs](https://developers.hubspot.com/docs/api/oauth-quickstart-guide)
- **Pandas y Numpy:** Usados para manipulación eficiente de datos tabulares y vectores numéricos.

**Parámetros clave y su justificación:**
- `max_tokens=180`: Limita la longitud de la respuesta de la IA para evitar cortes y cumplir con los límites de HubSpot.
- `temperature=0.2`: Baja aleatoriedad para respuestas más consistentes y menos creativas, priorizando la acción concreta.
- `playbook_notes` limitado a 300 caracteres: Para asegurar que el prompt no exceda el límite de tokens y la IA tenga espacio para responder.
- Fragmentos FAISS: Se seleccionan 3 fragmentos de máximo 60 caracteres para dar contexto sin saturar el prompt.

-------------------------------------------------------------------------------
## 3. Definiciones y Conceptos Clave
-------------------------------------------------------------------------------
- **Embeddings:** Representaciones vectoriales de texto que capturan significado semántico. Permiten comparar similitud entre frases/documentos. [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **FAISS:** Biblioteca de Facebook AI Research para búsqueda eficiente de similitud entre vectores de alta dimensión.
- **Prompt Engineering:** Técnica de diseño de instrucciones para IA, buscando maximizar la relevancia y utilidad de la respuesta. [Prompt Engineering Guide](https://www.promptingguide.ai/)
- **OAuth2:** Protocolo estándar para autorización segura entre aplicaciones y servicios de terceros. [OAuth2](https://oauth.net/2/)

-------------------------------------------------------------------------------
## 4. Alternativas Consideradas y Descartadas
-------------------------------------------------------------------------------
- **Frameworks:** Flask y Django REST fueron considerados, pero FastAPI ofrece mejor rendimiento y asincronía.
- **Modelos de IA:** Cohere y Google PaLM fueron evaluados, pero OpenAI GPT-4 tiene mejor soporte en español y mayor flexibilidad.
- **Búsqueda tradicional:** Se descartó por su baja precisión semántica frente a FAISS.

-------------------------------------------------------------------------------
## 5. Supuestos, Limitaciones y Recomendaciones
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
## 6. Ejemplo de Uso
-------------------------------------------------------------------------------
**Entrada (notas de playbook):**
    El cliente está interesado pero considera que el precio es alto frente a la competencia. Solicita una justificación clara del valor agregado. La decisión debe tomarse antes de fin de mes porque necesitan implementar la solución cuanto antes. El área financiera está involucrada y pide una propuesta formal.

**Respuesta esperada:**
    1. Preparar presentación de valor agregado.
    2. Enviar propuesta formal antes de fin de mes.
    3. Coordinar reunión con área financiera.

-------------------------------------------------------------------------------
## 7. Referencias y Lecturas Recomendadas
-------------------------------------------------------------------------------
- FastAPI: https://fastapi.tiangolo.com/
- OpenAI API: https://platform.openai.com/docs/
- FAISS: https://github.com/facebookresearch/faiss
- HubSpot API: https://developers.hubspot.com/docs/api/overview
- OAuth2: https://oauth.net/2/
- Prompt Engineering: https://www.promptingguide.ai/

-------------------------------------------------------------------------------
## 8. Descripción de Archivos principales del desarrollo/
-------------------------------------------------------------------------------

### backend.py
**Propósito:** Archivo principal del backend. Implementa la API RESTful, la lógica de integración con HubSpot, la generación de sugerencias con IA y la búsqueda semántica con FAISS.

**Importancia:** Es el núcleo del sistema, donde se orquesta la interacción entre el CRM, la IA y la base de conocimiento.

---

### Dockerfile
**Propósito:** Define el entorno de ejecución del backend para facilitar su despliegue en contenedores Docker.

**Importancia:** Permite portabilidad, reproducibilidad y despliegue sencillo en la nube o servidores.

---

### embeddings_openai.pkl
**Propósito:** Archivo binario que almacena los embeddings (vectores numéricos) precalculados de los fragmentos de la base de conocimiento.

**Importancia:** Permite realizar búsquedas semánticas rápidas y eficientes usando FAISS, sin recalcular embeddings en cada consulta.

---

### mapping.csv
**Propósito:** Archivo CSV que mapea los índices de FAISS a los textos originales de los fragmentos del playbook.

**Importancia:** Esencial para traducir los resultados de la búsqueda vectorial (índices) a fragmentos de texto útiles para el usuario.

**Ejemplo:**
```
idx,sentencias
0,"Presentar propuesta de valor al cliente."
1,"Solicitar feedback sobre la reunión anterior."
2,"Enviar documentación técnica requerida."
```

---

### playbook_index.faiss
**Propósito:** Archivo binario que contiene el índice vectorial FAISS para búsqueda de similitud semántica.

**Importancia:** Permite encontrar los fragmentos más relevantes para un contexto dado en tiempo real.

---

### Procfile
**Propósito:** Archivo de configuración para plataformas de despliegue como Heroku o Railway.

**Importancia:** Facilita el despliegue automatizado y la integración continua.

**Ejemplo:**
```
web: uvicorn backend:app --host=0.0.0.0 --port=8000
```

---

### README.md
**Propósito:** Documento de referencia para desarrolladores y evaluadores.

**Importancia:** Facilita la comprensión global del proyecto y su evaluación académica o profesional.

---

### Recolección y Organización de la Base de Conocimiento.ipynb
**Propósito:** Notebook de Jupyter usado para recolectar, limpiar y organizar los fragmentos de la base de conocimiento.

**Importancia:** Permite reproducir y auditar el proceso de construcción de la base de conocimiento.

---

### requirements.txt
**Propósito:** Lista de dependencias de Python necesarias para ejecutar el backend.

**Importancia:** Garantiza que el entorno de ejecución sea consistente y reproducible.

**Ejemplo:**
```
fastapi==0.95.2
openai==1.3.5
faiss-cpu==1.7.4
pandas==1.5.3
python-dotenv==1.0.0
```

---

### runtime.txt
**Propósito:** Especifica la versión de Python a usar en plataformas de despliegue.

**Importancia:** Evita incompatibilidades de versiones en el entorno de producción.

**Ejemplo:**
```
python-3.10.4
```

---

### sentencias.csv
**Propósito:** Archivo CSV que contiene los fragmentos de texto del playbook de ventas.

**Importancia:** Es la base de conocimiento sobre la que se fundamentan las recomendaciones de la IA.

**Ejemplo:**
```
sentencias
"Presentar propuesta de valor al cliente."
"Solicitar feedback sobre la reunión anterior."
"Enviar documentación técnica requerida."
```

---

### EDA (data).ipynb
**Propósito:** Notebook de Jupyter para Análisis Exploratorio de Datos (EDA) avanzado de la base de conocimiento. Realiza análisis estadístico multidimensional, evaluación de calidad para NLP y optimización de parámetros del sistema.

**Funcionalidades principales:**
- **Análisis básico**: Estadísticas descriptivas, distribuciones, frecuencias de palabras
- **Análisis avanzado**: Detección de outliers, diversidad léxica (TTR), correlaciones entre variables
- **Visualizaciones profesionales**: Gráficos multivariados, matrices de correlación, Q-Q plots

**Importancia académica:** Proporciona evidencia empírica robusta sobre la calidad del corpus, justificando cada parámetro del sistema con métricas objetivas. Las visualizaciones y métricas generadas constituyen material de alta calidad para presentaciones académicas y documentos de tesis de maestría.

**Dependencias específicas:** wordcloud, nltk, seaborn, matplotlib, spacy (modelo es_core_news_sm), scipy (para pruebas estadísticas)

**Salidas generadas:**

*Análisis básico:*
- `distribucion_longitud.png`: Histograma con líneas de referencia y estadísticas
- `palabras_comunes.png`: Gráfico de barras con valores y filtrado de stopwords
- `wordcloud.png`: Visualización semántica del corpus con configuración profesional
- `estadisticas_documentos.csv`: Métricas estadísticas detalladas por documento

*Análisis avanzado:*
- `analisis_avanzado_longitudes.png`: Panel de 4 gráficos (boxplots, densidad, Q-Q plot, outliers)
- `matriz_correlaciones.png`: Heatmap de correlaciones entre variables textuales
- `analisis_calidad_nlp.png`: Panel de 6 métricas específicas para tareas de NLP
- `metricas_avanzadas.json`: Todas las métricas calculadas en formato estructurado


