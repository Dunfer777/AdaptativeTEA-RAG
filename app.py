import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 0. CONFIGURACIÓN DE PÁGINA Y ENTORNO
st.set_page_config(
    page_title="AdaptativeTEA-RAG | Plataforma Pedagógica",
    page_icon="🧩",
    layout="wide"
)

load_dotenv()

# Cargar base de datos vectorial y embeddings en caché para mayor velocidad
@st.cache_resource
def iniciar_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

retriever = iniciar_vectorstore()

# Filtro ético y de seguridad para detección de crisis
PALABRAS_CRISIS = ["autolesion", "golpearse", "crisis severa", "agresion", "meltdown", "colapso"]
def evaluar_seguridad_crisis(texto):
    for palabra in PALABRAS_CRISIS:
        if palabra in texto.lower():
            return True
    return False

# INTERFAZ GRÁFICA (STREAMLIT)
st.title("🧩 AdaptativeTEA-RAG")
st.caption("Plataforma de Recomendación Pedagógica Inclusiva basada en RAG")
st.markdown("---")

# BARRA LATERAL: Formulario de Entrada de Datos del Estudiante
with st.sidebar:
    st.header("📋 Perfil del Estudiante")
    nombre = st.text_input("Nombre del estudiante", value="Lucas")
    edad = st.number_input("Edad", min_value=3, max_value=18, value=8)
    grado = st.text_input("Grado escolar", value="3° Primaria")
    nivel_tea = st.selectbox(
        "Nivel de soporte TEA",
        ["Nivel 1 (Requiere apoyo)", "Nivel 2 (Requiere apoyo sustancial)", "Nivel 3 (Requiere apoyo muy sustancial)"]
    )
    interes_especial = st.text_input("Interés especial / Motivador", value="Dinosaurios")
    estilo_aprendizaje = st.selectbox(
        "Estilo de aprendizaje principal",
        ["Visual / Manipulativo", "Auditivo / Secuencial", "Kinestésico / Vivencial"]
    )
    sensibilidades = st.text_area(
        "Sensibilidades o detonantes",
        value="Ruido fuerte e instrucciones verbales largas"
    )
    antecedentes_medicos = st.text_area(
        "Antecedentes / Adaptaciones de Salud",
        value="Hipersensibilidad auditiva, requiere pausas sensoriales cada 15 min."
    )
    
    st.markdown("---")
    # Interruptor para no gastar créditos durante las pruebas
    modo_prueba = st.checkbox("🧪 Modo Pruebas (Sin gastar saldo API)", value=True)

# ÁREA PRINCIPAL: Consulta del Docente
st.header("💬 Requerimiento Pedagógico")
consulta_docente = st.text_area(
    "Ingrese la duda o tema a adaptar para la clase:",
    value="Necesito una actividad didáctica para enseñarle sumas de dos dígitos con llevadas.",
    height=100
)

if st.button("🚀 Generar Plan Adaptado", type="primary"):
    if not consulta_docente.strip():
        st.warning("Por favor, ingrese un requerimiento o consulta.")
    elif evaluar_seguridad_crisis(consulta_docente):
        st.error("⚠️ **[ALERTA DE SEGURIDAD Y SALUD MENTAL DETECTADA]**")
        st.warning(
            "Se ha detectado un reporte de comportamiento de crisis o meltdown. "
            "La herramienta de IA no sustituye la intervención médica ni psicológica. "
            "**Acción recomendada:** Derive el caso inmediatamente al departamento de Psicología de la institución."
        )
    else:
        with st.spinner("Procesando información y buscando en manuales pedagógicos..."):
            if modo_prueba:
                # SIMULACIÓN LOCAL (0 Créditos gastados de OpenAI)
                time.sleep(1)  # Simula el tiempo de procesamiento RAG
                
                respuesta_simulada = f"""
### 🧪 [MODO PRUEBA ACTIVADO - RESPUESTA SIMULADA]

**Estudiante:** {nombre} ({edad} años, {grado})  
**Enfoque de Adaptación:** {interes_especial} | Estilo {estilo_aprendizaje}

---

#### 1. Actividad Didáctica Adaptada (20 minutos)
**Título:** *"La cueva de los {interes_especial} y la suma de unidades/decenas"*
- **Inicio (5 min):** Presentar a {nombre} un tablero visual donde las decenas son "cuevas" y las unidades son "huevos de dinosaurio".
- **Desarrollo (10 min):** Al sumar dos números que superan 10 unidades, {nombre} agrupa 10 huevos y los "convierte" físicamente en una cueva (llevada) que se desplaza a la columna adyacente.
- **Cierre (5 min):** Verificación con apoyo de pictogramas del resultado final.

#### 2. Apoyos Visuales y Manipulativos Recomendados
- Fichas manipulables de {interes_especial} divididas por colores (Azul = Unidades, Rojo = Decenas).
- Guía de pasos visuales plastificada sobre el escritorio.

#### 3. Pautas de Autorregulación Sensorial
- **Pauta 1 (Entorno):** {antecedentes_medicos}.
- **Pauta 2 (Prevención):** Si se detecta tensión por {sensibilidades}, realizar una pausa activa de 2 minutos con estímulo táctil de textura suave antes de continuar.

*(Filtro de seguridad validado exitosamente. No se realizaron llamadas cobradas a OpenAI).*
"""
                st.success("¡Plan Pedagógico Generado con Éxito (Simulación de Prueba)!")
                st.markdown("### 📄 Propuesta Adaptada")
                st.markdown(respuesta_simulada)
                
            else:
                # LLAMADA REAL A LA API DE OPENAI (Requiere crédito disponible)
                template_prompt = f"""Eres un Asistente Especializado en Pedagogía Inclusiva para niños con TEA.
Usa la siguiente información extraída de los manuales pedagógicos indexados para fundamentar tu respuesta:

{{context}}

DATOS DEL ESTUDIANTE:
- Nombre: {nombre} ({edad} años, {grado})
- Nivel TEA: {nivel_tea}
- Interés Especial: {interes_especial}
- Estilo de Aprendizaje: {estilo_aprendizaje}
- Sensibilidades / Detonantes: {sensibilidades}
- Antecedentes / Salud: {antecedentes_medicos}

CONSULTA DEL DOCENTE:
{{question}}

INSTRUCCIONES DE RESPUESTA:
1. Diseña una actividad didáctica adaptada de 20 minutos basada en su interés especial.
2. Especifica los apoyos visuales o manipulativos recomendados.
3. Incluye 2 pautas de autorregulación sensorial para prevenir sobrecargas en el aula.
"""
                PROMPT = PromptTemplate(
                    template=template_prompt,
                    input_variables=["context", "question"]
                )

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT}
                )

                respuesta = qa_chain.run(consulta_docente)

                st.success("¡Plan Pedagógico Generado con Éxito!")
                st.markdown("### 📄 Propuesta Adaptada")
                st.write(respuesta)
                