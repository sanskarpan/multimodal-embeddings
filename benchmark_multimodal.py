#!/usr/bin/env python3
"""
Benchmark multimodal embeddings: CLIP vs OpenAI
Evaluates performance using Precision@K, Recall@K, mAP, Hit Rate@K
"""

import os
import sys
import csv
import time
import json
import base64
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


class MultimodalBenchmark:
    """Benchmark multimodal embeddings"""
    
    def __init__(self, weaviate_url="http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.client: Optional[weaviate.WeaviateClient] = None
        self.dataset = []
        self.ground_truth_queries = []
        
    def connect(self):
        """Connect to Weaviate"""
        print_info(f"Connecting to Weaviate at {self.weaviate_url}...")
        
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )
        
        if self.client.is_ready():
            print_success("Connected to Weaviate")
        else:
            print_error("Failed to connect to Weaviate")
            sys.exit(1)
    
    def load_dataset(self, csv_path: str):
        """Load dataset from CSV"""
        print_info(f"Loading dataset from {csv_path}...")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.dataset = list(reader)
        
        # Load base64 encoded images
        for item in self.dataset:
            image_path = item['image_path']
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    item['image_base64'] = base64.b64encode(img_file.read()).decode()
            else:
                print_warning(f"Image not found: {image_path}")
                item['image_base64'] = None
        
        print_success(f"Loaded {len(self.dataset)} items from dataset")
        
        # Print dataset statistics
        categories = {}
        for item in self.dataset:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print_info("Dataset distribution:")
        for cat, count in categories.items():
            print(f"  • {cat}: {count} items")
    
    def load_ground_truth(self, queries_path: str):
        """Load ground truth queries"""
        print_info(f"Loading ground truth from {queries_path}...")
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse relevant_ids
                relevant_ids = [int(x.strip()) for x in row['relevant_ids'].split(',') if x.strip()]
                
                self.ground_truth_queries.append({
                    'query_id': row['query_id'],
                    'query_text': row['query_text'],
                    'query_category': row['query_category'],
                    'relevant_ids': relevant_ids,
                    'total_relevant': int(row['total_relevant'])
                })
        
        print_success(f"Loaded {len(self.ground_truth_queries)} ground truth queries")
    
    def create_collection_clip(self, collection_name="MultimodalCLIP"):
        """Create collection with CLIP embeddings"""
        print_info(f"Creating CLIP collection: {collection_name}")
        
        # Delete if exists
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
            print_info(f"Deleted existing collection: {collection_name}")
        
        # Create collection
        collection = self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.multi2vec_clip(
                text_fields=["description"],
                image_fields=["image"]
            ),
            properties=[
                Property(name="item_id", data_type=DataType.INT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="image", data_type=DataType.BLOB),
                Property(name="filename", data_type=DataType.TEXT),
            ]
        )
        
        print_success(f"Created CLIP collection: {collection_name}")
        return collection
    
    def create_collection_openai(self, collection_name="MultimodalOpenAI"):
        """Create collection with OpenAI text embeddings"""
        print_info(f"Creating OpenAI collection: {collection_name}")
        
        # Note: OpenAI doesn't have native multimodal embeddings yet like CLIP
        # We'll use text embeddings on the descriptions
        # For true multimodal with OpenAI, you'd need to use GPT-4V to generate
        # descriptions and then embed those, or wait for multimodal embedding API
        
        # Delete if exists
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
            print_info(f"Deleted existing collection: {collection_name}")
        
        # Create collection
        collection = self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small"
            ),
            properties=[
                Property(name="item_id", data_type=DataType.INT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
            ]
        )
        
        print_success(f"Created OpenAI collection: {collection_name}")
        return collection
    
    def insert_data_clip(self, collection_name="MultimodalCLIP"):
        """Insert data into CLIP collection"""
        print_info(f"Inserting {len(self.dataset)} items into {collection_name}...")
        
        collection = self.client.collections.get(collection_name)
        
        inserted = 0
        failed = 0
        
        with collection.batch.dynamic() as batch:
            for item in tqdm(self.dataset, desc="Inserting CLIP data"):
                try:
                    if item.get('image_base64'):
                        batch.add_object(
                            properties={
                                "item_id": int(item['id']),
                                "description": item['description'],
                                "category": item['category'],
                                "image": item['image_base64'],
                                "filename": item['filename'],
                            }
                        )
                        inserted += 1
                    else:
                        failed += 1
                except Exception as e:
                    print_warning(f"Failed to insert item {item.get('id')}: {e}")
                    failed += 1
        
        print_success(f"Inserted {inserted} items into {collection_name}")
        if failed > 0:
            print_warning(f"Failed to insert {failed} items")
        
        # Verify count
        response = collection.aggregate.over_all(total_count=True)
        print_info(f"Total items in {collection_name}: {response.total_count}")
    
    def insert_data_openai(self, collection_name="MultimodalOpenAI"):
        """Insert data into OpenAI collection"""
        print_info(f"Inserting {len(self.dataset)} items into {collection_name}...")
        
        collection = self.client.collections.get(collection_name)
        
        inserted = 0
        failed = 0
        
        with collection.batch.dynamic() as batch:
            for item in tqdm(self.dataset, desc="Inserting OpenAI data"):
                try:
                    batch.add_object(
                        properties={
                            "item_id": int(item['id']),
                            "description": item['description'],
                            "category": item['category'],
                            "filename": item['filename'],
                        }
                    )
                    inserted += 1
                except Exception as e:
                    print_warning(f"Failed to insert item {item.get('id')}: {e}")
                    failed += 1
        
        print_success(f"Inserted {inserted} items into {collection_name}")
        if failed > 0:
            print_warning(f"Failed to insert {failed} items")
        
        # Verify count
        response = collection.aggregate.over_all(total_count=True)
        print_info(f"Total items in {collection_name}: {response.total_count}")
    
    def search_and_evaluate(self, collection_name: str, k_values=[1, 3, 5, 10]):
        """Search and evaluate using ground truth queries"""
        print_header(f"Evaluating {collection_name}")
        
        collection = self.client.collections.get(collection_name)
        
        all_results = []
        query_times = []
        
        for query in tqdm(self.ground_truth_queries, desc=f"Running queries on {collection_name}"):
            start_time = time.time()
            
            # Search with max K
            max_k = max(k_values)
            try:
                response = collection.query.near_text(
                    query=query['query_text'],
                    limit=max_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Extract retrieved item IDs
                retrieved_ids = [obj.properties['item_id'] for obj in response.objects]
                
                # Store results
                all_results.append({
                    'query': query,
                    'retrieved_ids': retrieved_ids,
                    'query_time': query_time
                })
                
            except Exception as e:
                print_error(f"Query failed: {e}")
                all_results.append({
                    'query': query,
                    'retrieved_ids': [],
                    'query_time': 0
                })
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_results, k_values)
        
        # Add timing information
        metrics['avg_query_time'] = np.mean(query_times) if query_times else 0
        metrics['total_queries'] = len(self.ground_truth_queries)
        
        return metrics, all_results
    
    def calculate_metrics(self, results: List[Dict], k_values: List[int]) -> Dict:
        """Calculate evaluation metrics"""
        
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'hit_rate_at_k': {},
            'map': 0
        }
        
        all_ap = []  # Average Precision for each query
        
        for k in k_values:
            precisions = []
            recalls = []
            hits = []
            
            for result in results:
                query = result['query']
                retrieved = result['retrieved_ids'][:k]
                relevant = set(query['relevant_ids'])
                
                if len(retrieved) == 0:
                    precisions.append(0)
                    recalls.append(0)
                    hits.append(0)
                    continue
                
                # Precision@K
                true_positives = len([r for r in retrieved if r in relevant])
                precision = true_positives / len(retrieved) if len(retrieved) > 0 else 0
                precisions.append(precision)
                
                # Recall@K
                recall = true_positives / len(relevant) if len(relevant) > 0 else 0
                recalls.append(recall)
                
                # Hit Rate@K
                hit = 1 if true_positives > 0 else 0
                hits.append(hit)
            
            metrics['precision_at_k'][k] = np.mean(precisions) if precisions else 0
            metrics['recall_at_k'][k] = np.mean(recalls) if recalls else 0
            metrics['hit_rate_at_k'][k] = np.mean(hits) if hits else 0
        
        # Calculate mAP (Mean Average Precision)
        for result in results:
            query = result['query']
            retrieved = result['retrieved_ids']
            relevant = set(query['relevant_ids'])
            
            if len(relevant) == 0:
                continue
            
            # Calculate Average Precision for this query
            precisions_at_k = []
            relevant_found = 0
            
            for i, item_id in enumerate(retrieved, 1):
                if item_id in relevant:
                    relevant_found += 1
                    precision_at_i = relevant_found / i
                    precisions_at_k.append(precision_at_i)
            
            if len(precisions_at_k) > 0:
                ap = np.mean(precisions_at_k)
            else:
                ap = 0
            
            all_ap.append(ap)
        
        metrics['map'] = np.mean(all_ap) if all_ap else 0
        
        return metrics
    
    def print_metrics(self, model_name: str, metrics: Dict):
        """Print metrics in a formatted way"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}📊 {model_name} Results:{Colors.END}")
        print(f"{Colors.CYAN}{'─'*70}{Colors.END}")
        
        # Precision@K
        print(f"\n{Colors.BOLD}Precision@K:{Colors.END}")
        for k, value in metrics['precision_at_k'].items():
            print(f"  P@{k:2d} = {value:.4f}")
        
        # Recall@K
        print(f"\n{Colors.BOLD}Recall@K:{Colors.END}")
        for k, value in metrics['recall_at_k'].items():
            print(f"  R@{k:2d} = {value:.4f}")
        
        # Hit Rate@K
        print(f"\n{Colors.BOLD}Hit Rate@K:{Colors.END}")
        for k, value in metrics['hit_rate_at_k'].items():
            print(f"  HR@{k:2d} = {value:.4f}")
        
        # mAP
        print(f"\n{Colors.BOLD}Mean Average Precision (mAP):{Colors.END}")
        print(f"  mAP = {metrics['map']:.4f}")
        
        # Timing
        print(f"\n{Colors.BOLD}Performance:{Colors.END}")
        print(f"  Avg Query Time: {metrics['avg_query_time']*1000:.2f} ms")
        print(f"  Total Queries: {metrics['total_queries']}")
    
    def compare_models(self, clip_metrics: Dict, openai_metrics: Dict):
        """Compare two models side by side"""
        print_header("Model Comparison")
        
        # Create comparison table
        print(f"\n{Colors.BOLD}{'Metric':<25} {'CLIP':<15} {'OpenAI':<15} {'Winner':<15}{Colors.END}")
        print(f"{Colors.CYAN}{'─'*70}{Colors.END}")
        
        # Compare Precision@K
        for k in sorted(clip_metrics['precision_at_k'].keys()):
            clip_val = clip_metrics['precision_at_k'][k]
            openai_val = openai_metrics['precision_at_k'][k]
            winner = "CLIP" if clip_val > openai_val else "OpenAI" if openai_val > clip_val else "Tie"
            winner_color = Colors.GREEN if winner != "Tie" else Colors.YELLOW
            
            print(f"Precision@{k:<2d}            {clip_val:<15.4f} {openai_val:<15.4f} {winner_color}{winner}{Colors.END}")
        
        # Compare Recall@K
        for k in sorted(clip_metrics['recall_at_k'].keys()):
            clip_val = clip_metrics['recall_at_k'][k]
            openai_val = openai_metrics['recall_at_k'][k]
            winner = "CLIP" if clip_val > openai_val else "OpenAI" if openai_val > clip_val else "Tie"
            winner_color = Colors.GREEN if winner != "Tie" else Colors.YELLOW
            
            print(f"Recall@{k:<2d}               {clip_val:<15.4f} {openai_val:<15.4f} {winner_color}{winner}{Colors.END}")
        
        # Compare Hit Rate@K
        for k in sorted(clip_metrics['hit_rate_at_k'].keys()):
            clip_val = clip_metrics['hit_rate_at_k'][k]
            openai_val = openai_metrics['hit_rate_at_k'][k]
            winner = "CLIP" if clip_val > openai_val else "OpenAI" if openai_val > clip_val else "Tie"
            winner_color = Colors.GREEN if winner != "Tie" else Colors.YELLOW
            
            print(f"Hit Rate@{k:<2d}            {clip_val:<15.4f} {openai_val:<15.4f} {winner_color}{winner}{Colors.END}")
        
        # Compare mAP
        clip_map = clip_metrics['map']
        openai_map = openai_metrics['map']
        winner = "CLIP" if clip_map > openai_map else "OpenAI" if openai_map > clip_map else "Tie"
        winner_color = Colors.GREEN if winner != "Tie" else Colors.YELLOW
        print(f"mAP                      {clip_map:<15.4f} {openai_map:<15.4f} {winner_color}{winner}{Colors.END}")
        
        # Compare query time
        clip_time = clip_metrics['avg_query_time'] * 1000
        openai_time = openai_metrics['avg_query_time'] * 1000
        winner = "CLIP" if clip_time < openai_time else "OpenAI" if openai_time < clip_time else "Tie"
        winner_color = Colors.GREEN if winner != "Tie" else Colors.YELLOW
        print(f"Avg Query Time (ms)      {clip_time:<15.2f} {openai_time:<15.2f} {winner_color}{winner}{Colors.END}")
    
    def save_results(self, clip_metrics: Dict, openai_metrics: Dict, clip_results: List, openai_results: List, output_file="benchmark_results.json"):
        """Save benchmark results to file"""
        print_info(f"Saving results to {output_file}...")
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(self.dataset),
            'num_queries': len(self.ground_truth_queries),
            'clip': {
                'metrics': clip_metrics,
                'collection_name': 'MultimodalCLIP'
            },
            'openai': {
                'metrics': openai_metrics,
                'collection_name': 'MultimodalOpenAI'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print_success(f"Results saved to {output_file}")
    
    def cleanup(self):
        """Close connection"""
        if self.client:
            self.client.close()
            print_info("Disconnected from Weaviate")


def main():
    """Main benchmark execution"""
    
    print_header("Multimodal Embedding Benchmark: CLIP vs OpenAI")
    print_info("Evaluating Precision@K, Recall@K, mAP, Hit Rate@K\n")
    
    # Initialize benchmark
    benchmark = MultimodalBenchmark()
    
    try:
        # Connect to Weaviate
        benchmark.connect()
        
        # Load dataset
        dataset_csv = "synthetic_data/synthetic_dataset.csv"
        queries_csv = "synthetic_data/ground_truth_queries.csv"
        
        if not os.path.exists(dataset_csv):
            print_error(f"Dataset not found: {dataset_csv}")
            print_info("Please run: python generate_synthetic_dataset.py")
            sys.exit(1)
        
        benchmark.load_dataset(dataset_csv)
        benchmark.load_ground_truth(queries_csv)
        
        # K values for evaluation
        k_values = [1, 3, 5, 10]
        
        # ============ CLIP Benchmark ============
        print_header("Phase 1: CLIP Multimodal Embeddings")
        
        benchmark.create_collection_clip("MultimodalCLIP")
        benchmark.insert_data_clip("MultimodalCLIP")
        
        # Wait for indexing
        print_info("Waiting for CLIP indexing (3 seconds)...")
        time.sleep(3)
        
        clip_metrics, clip_results = benchmark.search_and_evaluate("MultimodalCLIP", k_values)
        benchmark.print_metrics("CLIP (multi2vec-clip)", clip_metrics)
        
        # ============ OpenAI Benchmark ============
        print_header("Phase 2: OpenAI Text Embeddings")
        
        benchmark.create_collection_openai("MultimodalOpenAI")
        benchmark.insert_data_openai("MultimodalOpenAI")
        
        # Wait for indexing
        print_info("Waiting for OpenAI indexing (3 seconds)...")
        time.sleep(3)
        
        openai_metrics, openai_results = benchmark.search_and_evaluate("MultimodalOpenAI", k_values)
        benchmark.print_metrics("OpenAI (text-embedding-3-small)", openai_metrics)
        
        # ============ Comparison ============
        benchmark.compare_models(clip_metrics, openai_metrics)
        
        # ============ Save Results ============
        benchmark.save_results(clip_metrics, openai_metrics, clip_results, openai_results)
        
        print_header("Benchmark Complete!")
        print_success("All evaluations finished successfully")
        
        # Print summary
        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Dataset Size: {len(benchmark.dataset)} items")
        print(f"  Ground Truth Queries: {len(benchmark.ground_truth_queries)}")
        print(f"  K Values Tested: {k_values}")
        print(f"  Results saved to: benchmark_results.json")
        
    except KeyboardInterrupt:
        print_warning("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()


