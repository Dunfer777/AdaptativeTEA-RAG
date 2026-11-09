import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 0. CARGAR CLAVE DE API DESDE VARIABLES DE ENTORNO (.env)
load_dotenv()

# 1. PERFILADOR DEL ESTUDIANTE CON TEA (6 a 12 años)
perfil_estudiante = {
    "nombre": "Lucas",
    "edad": 8,
    "grado": "3° Primaria",
    "nivel_tea": "Nivel 1 (Atención visual, requiere apoyo en comunicación verbal)",
    "interes_especial": "Dinosaurios",
    "estilo_aprendizaje": "Visual / Manipulación de objetos",
    "detonantes_sensoriales": "Ruido fuerte e instrucciones verbales largas",
    "tema_matematicas": "Sumas de dos dígitos con llevadas"
}

# 2. FILTRO DE SEGURIDAD Y ALERTA DE CRISIS (Eje Ético / Salud Mental)
PALABRAS_CRISIS = ["autolesion", "golpearse", "crisis severa", "agresion", "meltdown", "colapso"]

def evaluar_seguridad_crisis(consulta):
    for palabra in PALABRAS_CRISIS:
        if palabra in consulta.lower():
            return True
    return False

# 3. CARGAR EMBEDDINGS Y BASE VECTORIAL CHROMADB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 4. PLANTILLA DEL PROMPT CONECTADA AL CONTEXTO RAG
template_prompt = """Eres un Asistente Especializado en Pedagogía Inclusiva para niños con TEA.
Usa la siguiente información extraída de los manuales pedagógicos indexados para fundamentar tu respuesta:

{context}

DATOS DEL ESTUDIANTE:
- Nombre: """ + perfil_estudiante["nombre"] + """ (""" + str(perfil_estudiante["edad"]) + """ años, """ + perfil_estudiante["grado"] + """)
- Nivel TEA: """ + perfil_estudiante["nivel_tea"] + """
- Interés Especial: """ + perfil_estudiante["interes_especial"] + """
- Estilo de Aprendizaje: """ + perfil_estudiante["estilo_aprendizaje"] + """
- Sensibilidades: """ + perfil_estudiante["detonantes_sensoriales"] + """
- Tema Matemático: """ + perfil_estudiante["tema_matematicas"] + """

CONSULTA DEL DOCENTE:
{question}

INSTRUCCIONES DE RESPUESTA:
1. Diseña una actividad didáctica adaptada de 20 minutos basada en su interés especial.
2. Especifica los apoyos visuales o manipulativos recomendados.
3. Incluye 2 pautas de autorregulación sensorial para prevenir sobrecargas en el aula.
"""

PROMPT = PromptTemplate(
    template=template_prompt,
    input_variables=["context", "question"]
)

# 5. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    consulta_docente = "Necesito una actividad didáctica para enseñarle sumas de dos dígitos con llevadas."

    if evaluar_seguridad_crisis(consulta_docente):
        print("\n⚠️ [ALERTA DE SEGURIDAD DETECTADA]: Se reporta un comportamiento de crisis.")
        print("La IA no sustituye la intervención profesional. Por favor, derive el caso inmediatamente al departamento de Psicología de la institución.")
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )

        respuesta = qa_chain.run(consulta_docente)

        print("\n=================== PLAN PEDAGÓGICO ADAPTADO (TEA) ===================")
        print(respuesta)