#!/usr/bin/env python3
"""
Example script demonstrating Weaviate usage with OpenAI embeddings
"""

import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, MetadataQuery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Demonstrate basic Weaviate operations"""
    
    print("🚀 Connecting to Weaviate...")
    
    # Connect to Weaviate
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        }
    )
    
    try:
        if not client.is_ready():
            print("❌ Weaviate is not ready. Please start it with: ./setup_weaviate.sh start")
            return
        
        print("✅ Connected to Weaviate\n")
        
        # Example 1: Create a collection
        print("📦 Example 1: Creating a collection with OpenAI embeddings")
        collection_name = "ExampleArticles"
        
        # Delete if exists
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            print(f"   Deleted existing '{collection_name}' collection")
        
        # Create collection
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small"
            ),
            generative_config=Configure.Generative.openai(),
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
            ]
        )
        print(f"✅ Collection '{collection_name}' created\n")
        
        # Example 2: Insert data
        print("📝 Example 2: Inserting sample data")
        
        sample_articles = [
            {
                "title": "Getting Started with Python",
                "content": "Python is a versatile programming language perfect for beginners and experts alike.",
                "category": "Programming"
            },
            {
                "title": "Introduction to Docker",
                "content": "Docker containers provide a lightweight way to package and deploy applications.",
                "category": "DevOps"
            },
            {
                "title": "Vector Databases Explained",
                "content": "Vector databases enable semantic search and similarity matching using embeddings.",
                "category": "Database"
            },
            {
                "title": "Machine Learning Basics",
                "content": "Learn the fundamentals of machine learning including supervised and unsupervised learning.",
                "category": "AI"
            }
        ]
        
        with collection.batch.dynamic() as batch:
            for article in sample_articles:
                batch.add_object(properties=article)
        
        print(f"✅ Inserted {len(sample_articles)} articles\n")
        
        # Example 3: Semantic search
        print("🔍 Example 3: Semantic search")
        query = "containerization and deployment"
        print(f"   Query: '{query}'")
        
        response = collection.query.near_text(
            query=query,
            limit=2,
            return_metadata=MetadataQuery(distance=True)
        )
        
        print(f"   Found {len(response.objects)} results:")
        for i, obj in enumerate(response.objects, 1):
            print(f"   {i}. {obj.properties['title']}")
            print(f"      Category: {obj.properties['category']}")
            print(f"      Distance: {obj.metadata.distance:.4f}")
            print(f"      Content: {obj.properties['content'][:60]}...")
        print()
        
        # Example 4: Filtering
        print("🔎 Example 4: Search with filters")
        query = "learning"
        category_filter = "Programming"
        print(f"   Query: '{query}' + Filter: category='{category_filter}'")
        
        response = collection.query.near_text(
            query=query,
            filters=Filter.by_property("category").equal(category_filter),
            limit=5
        )
        
        print(f"   Found {len(response.objects)} results:")
        for obj in response.objects:
            print(f"   • {obj.properties['title']} ({obj.properties['category']})")
        print()
        
        # Example 5: Generative search (RAG)
        print("🤖 Example 5: Generative search (RAG)")
        query = "programming"
        prompt = "Based on this article, provide a one-sentence learning tip:"
        print(f"   Query: '{query}'")
        print(f"   Prompt: '{prompt}'")
        
        response = collection.generate.near_text(
            query=query,
            limit=2,
            single_prompt=prompt
        )
        
        print(f"   Generated responses:")
        for i, obj in enumerate(response.objects, 1):
            print(f"   {i}. Article: {obj.properties['title']}")
            print(f"      AI Tip: {obj.generated if obj.generated else 'No response'}")
        print()
        
        # Example 6: Aggregations
        print("📊 Example 6: Aggregations")
        response = collection.aggregate.over_all(total_count=True)
        print(f"   Total articles in collection: {response.total_count}\n")
        
        # Example 7: Get all unique categories
        print("📋 Example 7: Fetch all articles")
        response = collection.query.fetch_objects(limit=10)
        
        categories = set()
        for obj in response.objects:
            categories.add(obj.properties['category'])
        
        print(f"   Categories in database: {', '.join(sorted(categories))}\n")
        
        # Cleanup (optional)
        print("🧹 Cleanup: Deleting example collection")
        client.collections.delete(collection_name)
        print("✅ Example collection deleted\n")
        
        print("=" * 60)
        print("✨ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()
        print("\n👋 Disconnected from Weaviate")


if __name__ == "__main__":
    main()


