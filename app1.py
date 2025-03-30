import re
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from llama_index.core.storage.chat_store import SimpleChatStore
import os
import tempfile
import PyPDF2
from datetime import datetime
#New update 28 march
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained Legal-BERT model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Tokenize input text
text = "The payment terms specify that invoices must be paid within 30 days."
inputs = tokenizer(text, return_tensors="pt")

# Predict clause category
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
print("Predicted Clause Category:", predicted_class)

import fitz  # PyMuPDF

# Open PDF file
doc = fitz.open("legal_document.pdf")

# Iterate through pages and highlight specific text
for page in doc:
    text_instances = page.search_for("Payment Terms")
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)
        highlight.update()

# Save highlighted PDF
doc.save("highlighted_document.pdf")
#New update 28 march

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Chat store initialization
def load_chat_store():
    if os.path.exists("chats.json"):
        return SimpleChatStore.from_persist_path("chats.json")
    return SimpleChatStore()

chat_store = load_chat_store()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Extract system prompt from request
    system_prompt = request.form.get('system_prompt', "You are a legal assistant helping analyze Indian legal documents.Give response in the format of a Indian legal document in concise manner")

    # Save PDF temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(temp_path)

    # Extract text
    with open(temp_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])

    # Create session ID using filename + timestamp
    session_id = f"{pdf_file.filename}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Store PDF text as system message
    chat_store.add_message(
        session_id,
        {"role": "system", "content": f"{system_prompt}\nPDF Context:\n{text}"}
    )
    chat_store.persist(persist_path="chats.json")

    return jsonify({
        "session_id": session_id,
        "filename": pdf_file.filename
    })

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')

    if not session_id or not user_message:
        return jsonify({"error": "Missing parameters"}), 400

    # Get conversation history
    history = chat_store.get_messages(session_id)

    # Extract system prompt and context from history (system messages contain PDF text)
    system_context = next((msg["content"] for msg in history if msg["role"] == "system"), "You are a legal assistant helping analyze legal documents.")
    parts = system_context.split('\n', 1)
    system_prompt = parts[0] if len(parts) > 1 else system_context
    context = parts[1] if len(parts) > 1 else ""

    # Generate response with context
    prompt = f"{system_prompt}\n{context}\n\nUser Question: {user_message}\nAssistant Answer:"
    response = model.generate_content(prompt)

    # Remove Markdown characters from the response
    response_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response.text)  # Remove bold markdown

    # Store interaction
    chat_store.add_message(session_id, {"role": "user", "content": user_message})
    chat_store.add_message(session_id, {"role": "assistant", "content": response_text})  # Store cleaned response
    chat_store.persist(persist_path="chats.json")

    return jsonify({
        "response": response_text,
        "history": [{"role": msg["role"], "content": msg["content"]} for msg in chat_store.get_messages(session_id)[1:]]
    })

if __name__ == '__main__':
    app.run(debug=True)
