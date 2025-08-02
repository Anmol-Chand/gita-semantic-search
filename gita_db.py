from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

# Initialize vector DB and model
# client = chromadb.Client();
client = chromadb.PersistentClient(path="./gita_db", settings=Settings(anonymized_telemetry=False))

collection = client.create_collection(name="gita_shlokas_app_collection")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our sample shlokas with topics and chapter metadata
shlokas = [
    {
        "text": "You have the right to perform your prescribed duties, but never to the fruits of those actions.",
        "metadata": {"chapter": 2, "topic": "karma"}
    },
    {
        "text": "From anger arises delusion; from delusion, bewilderment of memory...",
        "metadata": {"chapter": 2, "topic": "anger"}
    },
    {
        "text": "A person who is not disturbed by happiness and distress is eligible for liberation.",
        "metadata": {"chapter": 2, "topic": "balance"}
    }
]

# Create sholka variables for adding in vector database # Prepare Data
shlokaIDs = [str(uuid.uuid4()) for _ in shlokas]
shlokaTexts = [s['text'] for s in shlokas]
shlokaMetaData = [s['metadata'] for s in shlokas]

shlokaEmbedding = model.encode(shlokaTexts)

# print(shlokaMetaData);

collection.add(
    ids=shlokaIDs,
    embeddings=shlokaEmbedding.tolist(),
    documents=shlokaTexts,
    metadatas=shlokaMetaData
)

print("✅ Shlokas added to ChromaDB!")