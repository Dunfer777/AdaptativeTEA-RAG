import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Cargar TODOS los PDFs dentro de la carpeta "Documentos pdf"
print("📄 Cargando documentos desde 'Documentos pdf'...")
loader = PyPDFDirectoryLoader("Documentos pdf")
docs = loader.load()
print(f"✅ Se cargaron {len(docs)} páginas de los PDFs.")

# 2. Dividir en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
fragmentos = splitter.split_documents(docs)
print(f"🧩 Se generaron {len(fragmentos)} fragmentos de texto.")

# 3. Crear embeddings (modelo multilingüe)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. Guardar en la base vectorial Chroma
db = Chroma.from_documents(
    documents=fragmentos, 
    embedding=embeddings, 
    persist_directory="db"
)

print("🚀 Base vectorial 'db' creada exitosamente.")