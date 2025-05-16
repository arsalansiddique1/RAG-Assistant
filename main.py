import os
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from langchain_openai.chat_models import ChatOpenAI
from config import settings
from pdf_loader import load_and_chunk_pdf
from vector_store import init_vector_store
from chain import get_qa_chain


# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_HISTORY = 20  # total messages
# Initialize Flask app
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Secret key for session management
app.config['SECRET_KEY'] = settings.FLASK_SECRET_KEY  # ensure this value is set in your environment


def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# One-time setup: vector store and QA chain
vectorstore = init_vector_store()
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.0
)

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 5})
qa_chain = get_qa_chain(llm, retriever)


@app.route("/")
def index():
    """
    Render the main chat UI.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle PDF upload, processing, and vector insertion.
    """
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Extract text chunks and add to vector store
    chunks = load_and_chunk_pdf(file_path)
    vectorstore.add_texts(
        chunks,
        metadatas=[{"source": filename}] * len(chunks)
    )

    return jsonify({"message": f"Uploaded and processed {filename}."}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or request.form
    question = data.get("question", "").strip()
    language = data.get("language", "en")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Map language codes
    lang_map  = {"en": "English", "fr": "French", "es": "Spanish"}
    lang_name = lang_map.get(language, "English")

    # Build prompt
    prompt_text = (
        f"Please answer the following question in {lang_name}:\n\n"
        f"{question}\n\n"
        f"Then ask the user in an assisting tone starting with 'Would' another "
        f"relevant follow‐up question in {lang_name}.\n\n"
    )

    # Retrieve session history
    chat_history = session.get('chat_history', [])

    # Prepare state with persisted history
    state = {
        "input": prompt_text,
        "chat_history": chat_history  # list of dicts with 'role' and 'content'
    }

    # Invoke chain (handles rephrasing + retrieval + answer)
    result = qa_chain.invoke(state)
    answer = result["answer"]

    # Update history
    chat_history.append({"role": "user", "content": prompt_text})
    chat_history.append({"role": "assistant", "content": answer})
    
    # Trim to the last MAX_HISTORY messages
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    session['chat_history'] = chat_history

    # Extract sources
    docs = result.get("context") or result.get("source_documents", [])
    sources = [doc.metadata.get("source") for doc in docs]

    # HTMX fallback
    if request.headers.get("HX-Request"):
        return render_template(
            "partials/message.html",
            answer=answer,
            sources=sources
        ), 200

    return jsonify({
        "answer":   answer,
        "sources":  sources,
        "language": f"in {lang_name}"
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
