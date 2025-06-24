import os
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai




# --- Configuraciones ---

# API KEY Google Gemini
API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Azure Blob Storage
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "chroma-db-container"
blob_name = "chroma_db.zip"
zip_path = "./chroma_db.zip"
extract_path = "./chroma_db/chroma_db"  # Ruta correcta para PersistentClient

# --- Funciones para Azure y ChromaDB ---

def descargar_blob():
    print("🔗 Conectando con Azure...")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print("📥 Descargando el ZIP...")
    with open(zip_path, "wb") as f:
        stream = blob_client.download_blob()
        for chunk in stream.chunks():
            f.write(chunk)
    print("✅ ZIP descargado correctamente.")

def descomprimir():
    print("📂 Descomprimiendo...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./chroma_db")
    print("✅ Descompresión completada.")

def listar_colecciones_y_chunk():
    print("🧠 Cargando cliente ChromaDB...")
    client = PersistentClient(path=extract_path)

    print("📋 Listando colecciones disponibles...")
    colecciones = client.list_collections()

    if not colecciones:
        print("⚠️ No hay colecciones en la base de datos.")
        return False

    print(f"✅ Se encontraron {len(colecciones)} colección(es):")
    for col in colecciones:
        print(f"🧱 - {col.name}")

    try:
        collection = client.get_collection("harrypotter_chunks")
        results = collection.query(query_texts=["prueba"], n_results=1)
        chunk = results.get("documents", [[""]])[0][0]
        print("✅ Primer chunk encontrado:")
        print(chunk)
        return True
    except Exception as e:
        print(f"⚠️ Error al consultar la colección: {e}")
        return False

# --- Setup Flask y modelos ---

app = Flask(__name__)
CORS(app)

genai.configure(api_key=API_KEY)
modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Client ChromaDB (lo inicializaremos más abajo, tras la descarga y descompresión)
client = None
collection = None

def generar_respuesta(prompt, contexto):
    full_prompt = (
        "Respondé la siguiente pregunta EXCLUSIVAMENTE con base en el contexto proporcionado. "
        "NO uses conocimiento externo, general o de sentido común. "
        "Si no hay suficiente información en el contexto, decí explícitamente que no podés responder con certeza. "
        "Adoptá el papel de un mago egocéntrico y dramático de Hogwarts, pero sin salirte del contenido del contexto. "
        "NO inventes datos, NO agregues información fuera del contexto.\n\n"
        f"### CONTEXTO:\n{contexto}\n\n"
        f"### PREGUNTA:\n{prompt}\n\n"
        "### RESPUESTA:"
    )
    try:
        respuesta = modelo_gemini.generate_content(full_prompt)
        return respuesta.text.strip()
    except Exception as e:
        print("❌ Error con Google AI:", e)
        return f"⚠️ Error con Google AI: {str(e)}"

@app.route("/query", methods=["POST"])
def query():
    global collection
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Falta el campo 'prompt'"}), 400

        embedding = embedding_model.encode(prompt).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=30)

        docs = results.get("documents", [[]])[0]
        contexto = "\n\n".join(docs)

        respuesta = generar_respuesta(prompt, contexto)

        return jsonify({
            "context": docs,
            "answer": respuesta
        })
    except Exception as e:
        print("❌ Error en /query:", e)
        return jsonify({"error": str(e)}), 500

# --- Main ---

if __name__ == "__main__":
    # Descargar y preparar ChromaDB si hace falta
    if not os.path.exists(zip_path):
        descargar_blob()
    else:
        print("📦 ZIP ya existe, omitiendo descarga.")

    if not os.path.exists(extract_path):
        descomprimir()
    else:
        print("📂 Carpeta ya descomprimida, omitiendo descompresión.")

    if not listar_colecciones_y_chunk():
        print("❌ No se pudo cargar la base de datos correctamente. Salir.")
        exit(1)

    # Inicializar cliente y colección global para usar en Flask
    client = PersistentClient(path=extract_path)
    collection = client.get_collection("harrypotter_chunks")

    print("🚀 Iniciando servidor Flask en http://localhost:5000")
    app.run(port=5000, debug=True)
