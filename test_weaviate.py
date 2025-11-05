#!/usr/bin/env python3
"""
Comprehensive test script for Weaviate with OpenAI and multimodal embeddings
"""

import os
import sys
import time
from typing import Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_section(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


class WeaviateTest:
    def __init__(self, url: str = "http://localhost:8080"):
        self.url = url
        self.client: Optional[weaviate.WeaviateClient] = None
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "total": 0
        }
        
    def connect(self) -> bool:
        """Connect to Weaviate instance"""
        try:
            print_info(f"Connecting to Weaviate at {self.url}...")
            
            # Connect to Weaviate
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )
            
            # Check if connected
            if self.client.is_ready():
                print_success("Connected to Weaviate successfully")
                return True
            else:
                print_error("Failed to connect to Weaviate")
                return False
                
        except Exception as e:
            print_error(f"Connection error: {str(e)}")
            return False
    
    def check_meta_info(self) -> bool:
        """Check Weaviate meta information and available modules"""
        try:
            print_info("Fetching Weaviate meta information...")
            meta = self.client.get_meta()
            
            print_success(f"Weaviate version: {meta.get('version', 'Unknown')}")
            
            # Check modules
            modules = meta.get('modules', {})
            print_info(f"Available modules: {len(modules)}")
            
            for module_name, module_info in modules.items():
                print(f"  • {module_name}")
            
            # Check for required modules
            required_modules = ['text2vec-openai', 'multi2vec-clip', 'generative-openai']
            missing_modules = []
            
            for module in required_modules:
                if module not in modules:
                    missing_modules.append(module)
            
            if missing_modules:
                print_error(f"Missing required modules: {', '.join(missing_modules)}")
                return False
            
            print_success("All required modules are available")
            return True
            
        except Exception as e:
            print_error(f"Failed to get meta info: {str(e)}")
            return False
    
    def create_text_collection(self) -> bool:
        """Create a collection with OpenAI text embeddings"""
        try:
            collection_name = "Articles"
            print_info(f"Creating collection '{collection_name}' with text2vec-openai...")
            
            # Delete if exists
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
                print_info(f"Deleted existing '{collection_name}' collection")
            
            # Create collection with OpenAI text vectorizer
            collection = self.client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                ),
                generative_config=Configure.Generative.openai(),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="author", data_type=DataType.TEXT),
                ]
            )
            
            print_success(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to create text collection: {str(e)}")
            return False
    
    def create_multimodal_collection(self) -> bool:
        """Create a collection with multimodal (CLIP) embeddings"""
        try:
            collection_name = "MultimodalContent"
            print_info(f"Creating collection '{collection_name}' with multi2vec-clip...")
            
            # Delete if exists
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
                print_info(f"Deleted existing '{collection_name}' collection")
            
            # Create collection with CLIP multimodal vectorizer
            # Specify which fields are text and which are images
            collection = self.client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.multi2vec_clip(
                    text_fields=["title", "description"],
                    image_fields=["image"]
                ),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="image", data_type=DataType.BLOB),
                    Property(name="category", data_type=DataType.TEXT),
                ]
            )
            
            print_success(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to create multimodal collection: {str(e)}")
            return False
    
    def insert_text_data(self) -> bool:
        """Insert sample text data"""
        try:
            collection_name = "Articles"
            print_info(f"Inserting sample data into '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            # Sample articles
            articles = [
                {
                    "title": "Introduction to Machine Learning",
                    "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                    "category": "Technology",
                    "author": "John Doe"
                },
                {
                    "title": "The Future of Electric Vehicles",
                    "content": "Electric vehicles are revolutionizing the automotive industry with their eco-friendly design and advanced technology features.",
                    "category": "Automotive",
                    "author": "Jane Smith"
                },
                {
                    "title": "Understanding Neural Networks",
                    "content": "Neural networks are computing systems inspired by biological neural networks. They are the foundation of deep learning algorithms.",
                    "category": "Technology",
                    "author": "Alice Johnson"
                },
                {
                    "title": "Climate Change and Renewable Energy",
                    "content": "Renewable energy sources like solar and wind power are crucial in combating climate change and reducing carbon emissions.",
                    "category": "Environment",
                    "author": "Bob Wilson"
                },
                {
                    "title": "The Art of Data Visualization",
                    "content": "Data visualization transforms complex datasets into visual representations, making it easier to identify patterns and insights.",
                    "category": "Data Science",
                    "author": "Carol Brown"
                }
            ]
            
            # Insert articles
            with collection.batch.dynamic() as batch:
                for article in articles:
                    batch.add_object(properties=article)
            
            print_success(f"Inserted {len(articles)} articles successfully")
            
            # Verify count
            response = collection.aggregate.over_all(total_count=True)
            print_info(f"Total objects in collection: {response.total_count}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to insert text data: {str(e)}")
            return False
    
    def test_text_search(self) -> bool:
        """Test semantic search on text data"""
        try:
            collection_name = "Articles"
            print_info(f"Testing semantic search on '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            # Test query
            query = "artificial intelligence and deep learning"
            print_info(f"Query: '{query}'")
            
            response = collection.query.near_text(
                query=query,
                limit=3,
                return_metadata=MetadataQuery(distance=True)
            )
            
            print_success(f"Found {len(response.objects)} results:")
            for i, obj in enumerate(response.objects, 1):
                print(f"\n  {i}. {obj.properties['title']}")
                print(f"     Category: {obj.properties['category']}")
                print(f"     Distance: {obj.metadata.distance:.4f}")
                print(f"     Content: {obj.properties['content'][:100]}...")
            
            return len(response.objects) > 0
            
        except Exception as e:
            print_error(f"Failed text search: {str(e)}")
            return False
    
    def test_generative_search(self) -> bool:
        """Test generative search (RAG) with OpenAI"""
        try:
            collection_name = "Articles"
            print_info(f"Testing generative search on '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            # Generative query
            query = "machine learning"
            prompt = "Summarize this article in one sentence:"
            
            print_info(f"Query: '{query}'")
            print_info(f"Prompt: '{prompt}'")
            
            response = collection.generate.near_text(
                query=query,
                limit=2,
                single_prompt=prompt
            )
            
            print_success(f"Generative search results:")
            for i, obj in enumerate(response.objects, 1):
                print(f"\n  {i}. {obj.properties['title']}")
                print(f"     Generated summary: {obj.generated}")
            
            return len(response.objects) > 0
            
        except Exception as e:
            print_error(f"Failed generative search: {str(e)}")
            print_info(f"Note: This requires OpenAI API access and may incur costs")
            return False
    
    def test_filters(self) -> bool:
        """Test filtering capabilities"""
        try:
            collection_name = "Articles"
            print_info(f"Testing filters on '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            from weaviate.classes.query import Filter
            
            # Filter by category
            response = collection.query.fetch_objects(
                filters=Filter.by_property("category").equal("Technology"),
                limit=10
            )
            
            print_success(f"Found {len(response.objects)} Technology articles:")
            for obj in response.objects:
                print(f"  • {obj.properties['title']}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed filter test: {str(e)}")
            return False
    
    def insert_multimodal_data(self) -> bool:
        """Insert sample multimodal data (text with placeholder for images)"""
        try:
            collection_name = "MultimodalContent"
            print_info(f"Inserting sample multimodal data into '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            # Sample multimodal content (without actual images for now)
            # In production, you would include base64 encoded images
            content_items = [
                {
                    "title": "Golden Retriever Puppy",
                    "description": "A cute golden retriever puppy playing in the garden with a red ball",
                    "category": "Animals"
                },
                {
                    "title": "Modern Architecture",
                    "description": "Contemporary glass and steel building with geometric design in urban setting",
                    "category": "Architecture"
                },
                {
                    "title": "Mountain Landscape",
                    "description": "Majestic snow-capped mountains with pine forest and blue sky at sunset",
                    "category": "Nature"
                }
            ]
            
            # Insert content
            with collection.batch.dynamic() as batch:
                for item in content_items:
                    batch.add_object(properties=item)
            
            print_success(f"Inserted {len(content_items)} multimodal items successfully")
            
            # Verify count
            response = collection.aggregate.over_all(total_count=True)
            print_info(f"Total objects in collection: {response.total_count}")
            
            return True
            
        except Exception as e:
            print_error(f"Failed to insert multimodal data: {str(e)}")
            return False
    
    def test_multimodal_search(self) -> bool:
        """Test semantic search on multimodal data"""
        try:
            collection_name = "MultimodalContent"
            print_info(f"Testing multimodal search on '{collection_name}'...")
            
            collection = self.client.collections.get(collection_name)
            
            # Test query
            query = "dog playing outside"
            print_info(f"Query: '{query}'")
            
            response = collection.query.near_text(
                query=query,
                limit=3,
                return_metadata=MetadataQuery(distance=True)
            )
            
            print_success(f"Found {len(response.objects)} results:")
            for i, obj in enumerate(response.objects, 1):
                print(f"\n  {i}. {obj.properties['title']}")
                print(f"     Category: {obj.properties['category']}")
                print(f"     Distance: {obj.metadata.distance:.4f}")
                print(f"     Description: {obj.properties['description']}")
            
            return len(response.objects) > 0
            
        except Exception as e:
            print_error(f"Failed multimodal search: {str(e)}")
            return False
    
    def cleanup(self) -> bool:
        """Optional: Clean up test collections"""
        try:
            print_info("Cleaning up test collections...")
            
            collections_to_delete = ["Articles", "MultimodalContent"]
            
            for collection_name in collections_to_delete:
                if self.client.collections.exists(collection_name):
                    self.client.collections.delete(collection_name)
                    print_info(f"Deleted collection '{collection_name}'")
            
            print_success("Cleanup completed")
            return True
            
        except Exception as e:
            print_error(f"Cleanup failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Weaviate"""
        if self.client:
            self.client.close()
            print_info("Disconnected from Weaviate")
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results"""
        self.test_results["total"] += 1
        print_section(f"Test: {test_name}")
        
        try:
            result = test_func()
            if result:
                self.test_results["passed"] += 1
                print_success(f"Test '{test_name}' PASSED\n")
            else:
                self.test_results["failed"] += 1
                print_error(f"Test '{test_name}' FAILED\n")
            return result
        except Exception as e:
            self.test_results["failed"] += 1
            print_error(f"Test '{test_name}' FAILED with exception: {str(e)}\n")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print_section("Test Summary")
        
        total = self.test_results["total"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        
        print(f"Total tests: {total}")
        print_success(f"Passed: {passed}")
        
        if failed > 0:
            print_error(f"Failed: {failed}")
        else:
            print_success(f"Failed: {failed}")
        
        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! {Colors.END}\n")
        else:
            print(f"\n{Colors.YELLOW}Some tests failed. Please check the output above.{Colors.END}\n")
        
        return failed == 0


def main():
    """Main test execution"""
    print_section("Weaviate Integration Test Suite")
    print_info("Testing Weaviate with OpenAI and Multimodal Embeddings\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY not found in environment variables")
        print_info("Please set it in your .env file")
        sys.exit(1)
    
    tester = WeaviateTest()
    
    try:
        # Connection test
        if not tester.connect():
            print_error("Failed to connect to Weaviate. Please ensure it's running.")
            print_info("Run: ./setup_weaviate.sh start")
            sys.exit(1)
        
        # Run all tests
        tester.run_test("Check Meta Information", tester.check_meta_info)
        tester.run_test("Create Text Collection with OpenAI", tester.create_text_collection)
        tester.run_test("Create Multimodal Collection with CLIP", tester.create_multimodal_collection)
        tester.run_test("Insert Text Data", tester.insert_text_data)
        tester.run_test("Test Semantic Text Search", tester.test_text_search)
        tester.run_test("Test Generative Search (RAG)", tester.test_generative_search)
        tester.run_test("Test Filters", tester.test_filters)
        tester.run_test("Insert Multimodal Data", tester.insert_multimodal_data)
        tester.run_test("Test Multimodal Search", tester.test_multimodal_search)
        
        # Print summary
        all_passed = tester.print_summary()
        
        # Optional cleanup (commented out by default to keep data for inspection)
        # tester.run_test("Cleanup Test Collections", tester.cleanup)
        
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print_info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        tester.disconnect()


if __name__ == "__main__":
    main()

