# Backend
from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Create Flask app
app = Flask(__name__)

# Load the same chromaDB used in gita_db.py 
# client =chromadb.Client()
client = chromadb.PersistentClient(path="./gita_db", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name="gita_shlokas_app_collection")

# Load the same embedding model we used for shlokas
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []

    if request.method == 'POST':
        user_query = request.form['query']

        print('❣️', user_query)
        query_embedding = model.encode(user_query)

        # Perform semantic search using ChromaDB
        query_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results= 3
        )
        # print('❣️', query_results)
        # Extract Documents and metadata from results
        documents = query_results['documents'][0]
        metadatas = query_results['metadatas'][0]

        # Pair each shloka with its metadata to display nicely
        results = [
            {"text": doc, "meta": meta}
            for doc, meta in zip(documents, metadatas)
        ]
    
    # Render the form and results using HTML template
    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
