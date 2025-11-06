# Quick Start Guide

## 🎯 Get Started in 3 Steps

### 1️⃣ Start Weaviate
```bash
./setup_weaviate.sh start
```

### 2️⃣ Activate Python Environment
```bash
source venv/bin/activate
```

### 3️⃣ Run Example or Tests
```bash
# Run the example script
python example_usage.py

# Or run comprehensive tests
python test_weaviate.py
```

---

## 📚 What's Included

```
Vectorizer/
├── docker-compose.yml       # Weaviate + CLIP containers
├── setup_weaviate.sh        # Management script
├── test_weaviate.py         # Comprehensive test suite
├── example_usage.py         # Usage examples
├── requirements.txt         # Python dependencies
├── .env                     # OpenAI API key
└── README.md               # Full documentation
```

---

## 🔧 Essential Commands

### Weaviate Management
```bash
./setup_weaviate.sh start      # Start containers
./setup_weaviate.sh stop       # Stop containers
./setup_weaviate.sh status     # Check status
./setup_weaviate.sh logs       # View logs
```

### Python Environment
```bash
source venv/bin/activate       # Activate venv
python test_weaviate.py        # Run tests
python example_usage.py        # Run examples
deactivate                     # Exit venv
```

---

## 🌐 Access Points

- **Weaviate REST API:** http://localhost:8080
- **Weaviate gRPC:** localhost:50051
- **Health Check:** http://localhost:8080/v1/.well-known/ready
- **Meta Info:** http://localhost:8080/v1/meta

---

## 💡 Quick Python Examples

### Connect to Weaviate
```python
import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)
```

### Create Collection with OpenAI
```python
from weaviate.classes.config import Configure, Property, DataType

collection = client.collections.create(
    name="MyDocs",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ]
)
```

### Insert Data
```python
# Single insert
collection.data.insert(
    properties={"title": "Hello", "content": "World"}
)

# Batch insert
with collection.batch.dynamic() as batch:
    for item in items:
        batch.add_object(properties=item)
```

### Semantic Search
```python
from weaviate.classes.query import MetadataQuery

response = collection.query.near_text(
    query="your search query",
    limit=5,
    return_metadata=MetadataQuery(distance=True)
)

for obj in response.objects:
    print(obj.properties['title'])
    print(f"Distance: {obj.metadata.distance}")
```

---

## 🎨 Features Available

✅ **Text Embeddings** - OpenAI text-embedding-3-small  
✅ **Multimodal Embeddings** - CLIP (text + images)  
✅ **Semantic Search** - Vector similarity search  
✅ **Filtering** - Combine vectors with filters  
✅ **RAG/Generative AI** - Generate text with OpenAI  
✅ **Batch Operations** - Efficient bulk inserts  
✅ **Persistent Storage** - Data saved in Docker volume  

---

## 🐛 Troubleshooting

### Weaviate won't start
```bash
docker ps  # Check if containers are running
./setup_weaviate.sh logs  # Check logs
./setup_weaviate.sh restart  # Try restarting
```

### Python connection errors
- Ensure Weaviate is running: `./setup_weaviate.sh status`
- Check OpenAI API key in `.env`
- Activate venv: `source venv/bin/activate`

### Tests failing
```bash
# Check Weaviate status
./setup_weaviate.sh status

# Restart Weaviate
./setup_weaviate.sh restart

# Wait 10 seconds, then run tests
python test_weaviate.py
```

---

## 📖 Learn More

- Full documentation: See `README.md`
- Test suite: Check `test_weaviate.py` for more examples
- Example code: Run `python example_usage.py`

---

## 🎓 Next Steps

1. **Explore Collections:**
   - Check the `Articles` collection in tests
   - Look at `MultimodalContent` for image support

2. **Try Custom Data:**
   - Modify `example_usage.py` with your data
   - Create new collections for your use case

3. **Advanced Features:**
   - Add authentication (production)
   - Configure backups
   - Optimize for your workload

4. **Integration:**
   - Connect from your application
   - Build a RAG chatbot
   - Create a semantic search engine

---

**Need Help?**
- Check `README.md` for detailed documentation
- Review `test_weaviate.py` for complete API examples
- Visit: https://weaviate.io/developers/weaviate


