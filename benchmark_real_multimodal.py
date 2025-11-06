#!/usr/bin/env python3
"""
Benchmark CLIP vs OpenAI multimodal embeddings with REAL images
Compares performance on actual image-text cross-retrieval
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import weaviate
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CLIP_COLLECTION = "MyrityProducts_CLIP_Real"
OPENAI_COLLECTION = "MyrityProducts_OpenAI_Real"
RESULTS_FILE = "benchmark_real_results.json"


@dataclass
class Item:
    """Item from Weaviate collection"""
    uuid: str
    handle: str
    title: Optional[str] = None
    caption: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class MetricsResult:
    """Evaluation metrics"""
    precision_at_k: float
    recall_at_k: float
    mean_avg_precision: float
    hit_rate_at_k: float
    mrr: float  # Mean Reciprocal Rank
    num_queries: int
    avg_query_time_ms: float = 0.0


def load_items_from_collection(client: weaviate.WeaviateClient, 
                                collection_name: str) -> List[Item]:
    """
    Load all items from collection (now they're all multimodal)
    """
    collection = client.collections.get(collection_name)
    
    items = []
    
    print(f"  Loading items from {collection_name}...")
    
    for obj in collection.iterator(
        return_properties=["handle", "title", "caption", "image_url"],
        include_vector=False
    ):
        props = obj.properties or {}
        
        item = Item(
            uuid=str(obj.uuid),
            handle=props.get("handle", ""),
            title=props.get("title"),
            caption=props.get("caption"),
            image_url=props.get("image_url")
        )
        items.append(item)
    
    print(f"    ✓ Loaded {len(items)} multimodal items")
    return items


def calculate_precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0 or not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for item in retrieved_k if item in relevant)
    return hits / len(retrieved_k)


def calculate_recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Calculate Recall@K"""
    if not relevant:
        return 0.0
    if k == 0 or not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for item in retrieved_k if item in relevant)
    return hits / len(relevant)


def calculate_average_precision(relevant: Set[str], retrieved: List[str]) -> float:
    """Calculate Average Precision"""
    if not relevant or not retrieved:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            score += precision_at_i
    
    return score / len(relevant) if relevant else 0.0


def calculate_hit_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """Calculate Hit@K (binary: 1 if any relevant in top-k, else 0)"""
    if k == 0 or not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    return 1.0 if any(item in relevant for item in retrieved_k) else 0.0


def calculate_reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    """Calculate Reciprocal Rank (1/rank of first relevant item)"""
    if not relevant or not retrieved:
        return 0.0
    
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    client: weaviate.WeaviateClient,
    collection_name: str,
    items: List[Item],
    k: int = 5
) -> MetricsResult:
    """
    Evaluate retrieval: each item queries for other items with same handle
    Since all objects are multimodal, this tests unified embedding quality
    """
    collection = client.collections.get(collection_name)
    
    # Build handle→items mapping
    handle_map = {}
    for item in items:
        if item.handle:
            if item.handle not in handle_map:
                handle_map[item.handle] = []
            handle_map[item.handle].append(item)
    
    # Filter to handles with multiple items (variants/duplicates)
    handles_with_variants = {h: items for h, items in handle_map.items() if len(items) > 1}
    
    if not handles_with_variants:
        print("  ⚠️  No product variants found for evaluation")
        return MetricsResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    
    precisions = []
    recalls = []
    avg_precisions = []
    hits = []
    rrs = []
    query_times = []
    
    # Use subset of items for queries to speed up
    query_items = []
    for handle, variants in handles_with_variants.items():
        query_items.extend(variants[:2])  # Max 2 queries per handle
    
    for query_item in tqdm(query_items, desc="  Queries"):
        # Get ground truth: other items with same handle
        same_handle_items = [it for it in items if it.handle == query_item.handle and it.uuid != query_item.uuid]
        
        if not same_handle_items:
            continue
        
        relevant_uuids = {it.uuid for it in same_handle_items}
        
        # Perform query
        start_time = time.perf_counter()
        try:
            response = collection.query.near_object(
                near_object=query_item.uuid,
                limit=k + 1,  # +1 to exclude self
                return_properties=["handle"],
                return_metadata=MetadataQuery(distance=True)
            )
        except Exception as e:
            print(f"\n  Error querying {query_item.uuid}: {e}")
            continue
        end_time = time.perf_counter()
        
        query_times.append((end_time - start_time) * 1000)  # ms
        
        # Extract retrieved UUIDs (excluding self)
        retrieved_uuids = [str(obj.uuid) for obj in response.objects if str(obj.uuid) != query_item.uuid][:k]
        
        # Calculate metrics
        precisions.append(calculate_precision_at_k(relevant_uuids, retrieved_uuids, k))
        recalls.append(calculate_recall_at_k(relevant_uuids, retrieved_uuids, k))
        avg_precisions.append(calculate_average_precision(relevant_uuids, retrieved_uuids))
        hits.append(calculate_hit_at_k(relevant_uuids, retrieved_uuids, k))
        rrs.append(calculate_reciprocal_rank(relevant_uuids, retrieved_uuids))
    
    num_queries = len(precisions)
    if num_queries == 0:
        return MetricsResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    
    return MetricsResult(
        precision_at_k=sum(precisions) / num_queries,
        recall_at_k=sum(recalls) / num_queries,
        mean_avg_precision=sum(avg_precisions) / num_queries,
        hit_rate_at_k=sum(hits) / num_queries,
        mrr=sum(rrs) / num_queries,
        num_queries=num_queries,
        avg_query_time_ms=sum(query_times) / num_queries
    )


def print_retrieval_examples(
    client: weaviate.WeaviateClient,
    collection_name: str,
    items: List[Item],
    k: int = 5,
    max_examples: int = 3
):
    """Print example retrievals"""
    collection = client.collections.get(collection_name)
    
    # Get items with variants for better examples
    handle_map = {}
    for item in items:
        if item.handle:
            if item.handle not in handle_map:
                handle_map[item.handle] = []
            handle_map[item.handle].append(item)
    
    example_items = []
    for handle, variants in handle_map.items():
        if len(variants) > 1:
            example_items.append(variants[0])
        if len(example_items) >= max_examples:
            break
    
    for idx, query_item in enumerate(example_items):
        print(f"\n  Example {idx+1}:")
        print(f"    Query: {query_item.title or query_item.caption or query_item.handle}")
        
        try:
            response = collection.query.near_object(
                near_object=query_item.uuid,
                limit=k + 1,
                return_properties=["handle", "title", "caption"],
                return_metadata=MetadataQuery(distance=True)
            )
            
            rank = 1
            for obj in response.objects:
                if str(obj.uuid) == query_item.uuid:
                    continue  # Skip self
                props = obj.properties or {}
                match = "✓" if props.get("handle") == query_item.handle else "✗"
                label = props.get("title") or props.get("caption") or props.get("handle", "")
                dist = obj.metadata.distance if obj.metadata else None
                print(f"      {rank}. {match} {label[:55]}... (dist: {dist:.3f})" if dist is not None else f"      {rank}. {match} {label[:55]}...")
                rank += 1
                if rank > k:
                    break
        
        except Exception as e:
            print(f"      Error: {e}")


def benchmark_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
    k: int = 5
) -> MetricsResult:
    """
    Benchmark a single collection with unified multimodal embeddings
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {collection_name}")
    print(f"{'='*80}")
    
    # Load all items
    items = load_items_from_collection(client, collection_name)
    
    if not items:
        print("  ✗ No items found!")
        return MetricsResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    
    # Evaluate retrieval
    print(f"\n{'─'*80}")
    print(f"MULTIMODAL RETRIEVAL EVALUATION")
    print(f"{'─'*80}")
    print(f"  Testing unified embeddings (text + image → similar products)")
    
    metrics = evaluate_retrieval(client, collection_name, items, k=k)
    
    print(f"\n  Results:")
    print(f"    Queries:            {metrics.num_queries}")
    print(f"    Precision@{k}:       {metrics.precision_at_k:.4f}")
    print(f"    Recall@{k}:          {metrics.recall_at_k:.4f}")
    print(f"    mAP:                {metrics.mean_avg_precision:.4f}")
    print(f"    Hit Rate@{k}:        {metrics.hit_rate_at_k:.4f}")
    print(f"    MRR:                {metrics.mrr:.4f}")
    print(f"    Avg Query Time:     {metrics.avg_query_time_ms:.2f}ms")
    
    print(f"\n  Examples (✓ = same product/variant):")
    print_retrieval_examples(client, collection_name, items, k=k, max_examples=5)
    
    return metrics


def print_comparison(clip_metrics: MetricsResult, openai_metrics: MetricsResult, k: int):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY (K={k})")
    print(f"{'='*80}")
    print("\nUNIFIED MULTIMODAL RETRIEVAL (Text + Image → Similar Products):")
    print(f"{'Metric':<25} {'CLIP':>12} {'OpenAI':>12} {'Winner':>12}")
    print("─" * 65)
    print(f"{'Precision@K':<25} {clip_metrics.precision_at_k:>12.4f} {openai_metrics.precision_at_k:>12.4f} {('OpenAI' if openai_metrics.precision_at_k > clip_metrics.precision_at_k else ('CLIP' if clip_metrics.precision_at_k > openai_metrics.precision_at_k else 'Tied')):>12}")
    print(f"{'Recall@K':<25} {clip_metrics.recall_at_k:>12.4f} {openai_metrics.recall_at_k:>12.4f} {('OpenAI' if openai_metrics.recall_at_k > clip_metrics.recall_at_k else ('CLIP' if clip_metrics.recall_at_k > openai_metrics.recall_at_k else 'Tied')):>12}")
    print(f"{'mAP':<25} {clip_metrics.mean_avg_precision:>12.4f} {openai_metrics.mean_avg_precision:>12.4f} {('OpenAI' if openai_metrics.mean_avg_precision > clip_metrics.mean_avg_precision else ('CLIP' if clip_metrics.mean_avg_precision > openai_metrics.mean_avg_precision else 'Tied')):>12}")
    print(f"{'Hit Rate@K':<25} {clip_metrics.hit_rate_at_k:>12.4f} {openai_metrics.hit_rate_at_k:>12.4f} {('OpenAI' if openai_metrics.hit_rate_at_k > clip_metrics.hit_rate_at_k else ('CLIP' if clip_metrics.hit_rate_at_k > openai_metrics.hit_rate_at_k else 'Tied')):>12}")
    print(f"{'MRR':<25} {clip_metrics.mrr:>12.4f} {openai_metrics.mrr:>12.4f} {('OpenAI' if openai_metrics.mrr > clip_metrics.mrr else ('CLIP' if clip_metrics.mrr > openai_metrics.mrr else 'Tied')):>12}")
    print(f"{'Avg Query Time (ms)':<25} {clip_metrics.avg_query_time_ms:>12.2f} {openai_metrics.avg_query_time_ms:>12.2f} {'CLIP ⚡' if clip_metrics.avg_query_time_ms < openai_metrics.avg_query_time_ms else ('OpenAI ⚡' if openai_metrics.avg_query_time_ms < clip_metrics.avg_query_time_ms else 'Tied'):>12}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CLIP vs OpenAI with real images")
    parser.add_argument("--k", type=int, default=5, help="K value for metrics")
    args = parser.parse_args()
    
    k = args.k
    
    print("="*80)
    print("REAL MULTIMODAL BENCHMARK - CLIP vs OpenAI")
    print("="*80)
    
    # Connect
    print("\nConnecting to Weaviate...")
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )
    
    try:
        if not client.is_ready():
            print("✗ Weaviate not ready!")
            sys.exit(1)
        print("✓ Connected")
        
        # Check collections exist
        if not client.collections.exists(CLIP_COLLECTION):
            print(f"\n✗ Collection '{CLIP_COLLECTION}' not found!")
            print(f"  Run: python load_clip_collection_v2.py --sample 50")
            sys.exit(1)
        
        if not client.collections.exists(OPENAI_COLLECTION):
            print(f"\n✗ Collection '{OPENAI_COLLECTION}' not found!")
            print(f"  Run: python load_openai_collection_v2.py --sample 50")
            sys.exit(1)
        
        # Benchmark both
        clip_metrics = benchmark_collection(client, CLIP_COLLECTION, k=k)
        openai_metrics = benchmark_collection(client, OPENAI_COLLECTION, k=k)
        
        # Print comparison
        print_comparison(clip_metrics, openai_metrics, k)
        
        # Save results
        all_results = {
            "CLIP": asdict(clip_metrics),
            "OpenAI": asdict(openai_metrics),
            "k": k,
            "note": "Unified multimodal embeddings (single object with text + image)"
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Results saved to {RESULTS_FILE}")
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()

