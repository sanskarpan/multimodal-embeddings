# Weaviate Vector Database Setup

This repository contains a complete setup for Weaviate vector database with OpenAI embeddings and multimodal support using CLIP.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.12+ with virtual environment
- OpenAI API key

### Setup

1. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

2. **Start Weaviate:**
```bash
./setup_weaviate.sh start
```

3. **Run tests:**
```bash
python test_weaviate.py
```

## 📋 Available Commands

The `setup_weaviate.sh` script provides several commands:

```bash
./setup_weaviate.sh start    # Start Weaviate containers
./setup_weaviate.sh stop     # Stop Weaviate containers
./setup_weaviate.sh restart  # Restart Weaviate containers
./setup_weaviate.sh status   # Show Weaviate status and meta information
./setup_weaviate.sh logs     # Show and follow Weaviate logs
./setup_weaviate.sh clean    # Stop and remove all data (WARNING: destructive!)
```

## 🏗️ Architecture

### Components

1. **Weaviate Core** (Port 8080)
   - Main vector database instance
   - RESTful API endpoint
   - gRPC endpoint (Port 50051)

2. **multi2vec-clip** Container
   - CLIP model for multimodal embeddings
   - Supports both text and image vectorization
   - Uses ViT-B-32 multilingual model

3. **Modules Enabled:**
   - `text2vec-openai` - OpenAI text embeddings
   - `multi2vec-clip` - Multimodal (text + image) embeddings
   - `generative-openai` - RAG/Generative AI capabilities

## 📦 Collections

### Text Collection (Articles)
Uses OpenAI's `text-embedding-3-small` model for semantic search on text data.

**Properties:**
- `title` (TEXT)
- `content` (TEXT)
- `category` (TEXT)
- `author` (TEXT)

**Example Usage:**
```python
import weaviate
from weaviate.classes.query import MetadataQuery

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    headers={"X-OpenAI-Api-Key": "your-api-key"}
)

# Get collection
articles = client.collections.get("Articles")

# Semantic search
response = articles.query.near_text(
    query="artificial intelligence",
    limit=5,
    return_metadata=MetadataQuery(distance=True)
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Distance: {obj.metadata.distance}")
```

### Multimodal Collection (MultimodalContent)
Uses CLIP model for searching across text and images.

**Properties:**
- `title` (TEXT)
- `description` (TEXT)
- `image` (BLOB) - Base64 encoded images
- `category` (TEXT)

**Example Usage:**
```python
# Search with text query (works on both text and images)
response = collection.query.near_text(
    query="dog playing outside",
    limit=3
)

# Search with image
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = collection.query.near_image(
    near_image=image_data,
    limit=3
)
```

## 🧪 Testing

The test suite (`test_weaviate.py`) includes:

1. ✅ Connection and meta information check
2. ✅ Text collection creation with OpenAI embeddings
3. ✅ Multimodal collection creation with CLIP
4. ✅ Data insertion (text and multimodal)
5. ✅ Semantic search on text
6. ✅ Generative search (RAG with OpenAI)
7. ✅ Filtering capabilities
8. ✅ Multimodal search

**Run tests:**
```bash
source venv/bin/activate
python test_weaviate.py
```

## 🔧 Configuration

### Environment Variables

Set in `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Docker Compose Configuration

Key settings in `docker-compose.yml`:

- **Persistence:** Data stored in Docker volume `weaviate_data`
- **Authentication:** Anonymous access enabled (disable in production!)
- **Ports:** 8080 (HTTP), 50051 (gRPC)
- **Modules:** OpenAI and CLIP vectorizers enabled

## 📊 API Endpoints

### Health Check
```bash
curl http://localhost:8080/v1/.well-known/ready
```

### Meta Information
```bash
curl http://localhost:8080/v1/meta | python3 -m json.tool
```

### List Collections
```bash
curl http://localhost:8080/v1/schema
```

## 🔍 Common Operations

### Creating a Collection with OpenAI Embeddings

```python
from weaviate.classes.config import Configure, Property, DataType

collection = client.collections.create(
    name="MyCollection",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    generative_config=Configure.Generative.openai(),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ]
)
```

### Creating a Collection with CLIP (Multimodal)

```python
collection = client.collections.create(
    name="MultimodalCollection",
    vectorizer_config=Configure.Vectorizer.multi2vec_clip(
        text_fields=["title", "description"],
        image_fields=["image"]
    ),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
        Property(name="image", data_type=DataType.BLOB),
    ]
)
```

### Inserting Data

```python
# Single insert
collection.data.insert(
    properties={
        "title": "Example Title",
        "content": "Example content here"
    }
)

# Batch insert
with collection.batch.dynamic() as batch:
    for item in data_items:
        batch.add_object(properties=item)
```

### Semantic Search

```python
from weaviate.classes.query import Filter, MetadataQuery

# Near text search
response = collection.query.near_text(
    query="search query",
    limit=10,
    return_metadata=MetadataQuery(distance=True, certainty=True)
)

# With filters
response = collection.query.near_text(
    query="search query",
    filters=Filter.by_property("category").equal("Technology"),
    limit=10
)
```

### Generative Search (RAG)

```python
response = collection.generate.near_text(
    query="machine learning",
    limit=5,
    single_prompt="Summarize this article in one sentence:"
)

for obj in response.objects:
    print(f"Title: {obj.properties['title']}")
    print(f"Generated: {obj.generated}")
```

## 🛠️ Troubleshooting

### Weaviate won't start
```bash
# Check Docker is running
docker info

# Check logs
./setup_weaviate.sh logs

# Restart everything
./setup_weaviate.sh restart
```

### Connection errors in Python
```python
# Ensure you're using the correct connection method
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

# Always close the client when done
client.close()
```

### OpenAI API errors
- Check your API key in `.env`
- Verify you have credits in your OpenAI account
- Check rate limits

### CLIP model errors
- Ensure the multi2vec-clip container is running: `docker ps`
- Check logs: `docker logs multi2vec-clip`
- The CLIP service needs time to initialize (30-60 seconds)

## 📈 Performance Tips

1. **Batch Operations:** Use batch inserts for large datasets
2. **Vector Cache:** Weaviate caches vectors in memory for fast retrieval
3. **Disk Space:** Monitor disk usage - vectors consume significant space
4. **OpenAI Rate Limits:** Be aware of OpenAI API rate limits
5. **CLIP Memory:** The CLIP model loads into RAM (~1-2GB)

## 🔒 Security Notes

⚠️ **Important for Production:**

1. Disable anonymous access in `docker-compose.yml`
2. Enable authentication (API keys or OIDC)
3. Use HTTPS/TLS for API communication
4. Restrict network access to Weaviate ports
5. Keep OpenAI API key secure (use secrets management)
6. Regular backups of the `weaviate_data` volume

## 📚 Additional Resources

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [CLIP Model](https://github.com/openai/CLIP)

## 🐛 Known Issues

1. **Deprecation Warning:** The `vectorizer_config` parameter shows a deprecation warning. This is expected with the current client version and will be updated in future releases.

2. **Generative Search Empty Results:** If generative search returns empty strings, check:
   - OpenAI API key is valid
   - You have sufficient credits
   - The prompt is properly formatted

## 📝 License

This setup is provided as-is for development and testing purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Weaviate logs
3. Consult official documentation
4. Check OpenAI API status

---

**Last Updated:** November 4, 2025
**Weaviate Version:** 1.27.3
**Python Client:** 4.17.0
**OpenAI Client:** 2.7.1


