#!/usr/bin/env python3
"""
Benchmark script for comparing CLIP vs OpenAI multimodal retrieval
Implements metrics from cross_retrieval_analysis.py:
- Precision@K
- Recall@K
- Mean Average Precision (mAP)
- Hit Rate@K
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TOP_K = 5  # Number of results to retrieve


@dataclass(frozen=True)
class Item:
    """Represents a product item (text or image)"""
    uuid: str
    handle: str
    modality: Optional[str]
    title: Optional[str]
    description: Optional[str]
    image_url: Optional[str]
    caption: Optional[str]


@dataclass(frozen=True)
class MetricsResult:
    """Evaluation metrics"""
    precision_at_k: float
    recall_at_k: float
    mean_avg_precision: float
    hit_rate_at_k: float
    num_queries: int
    avg_query_time_ms: float


def load_all_items(client: weaviate.WeaviateClient, collection_name: str) -> List[Item]:
    """Load all items from a collection"""
    collection = client.collections.get(collection_name)
    items: List[Item] = []
    
    ret_props = ["handle", "modality", "title", "description", "image_url", "caption"]
    
    for obj in collection.iterator(return_properties=ret_props):
        props = obj.properties or {}
        items.append(Item(
            uuid=str(obj.uuid),
            handle=props.get("handle", ""),
            modality=props.get("modality"),
            title=props.get("title"),
            description=props.get("description"),
            image_url=props.get("image_url"),
            caption=props.get("caption")
        ))
    
    return items


def near_modality(client: weaviate.WeaviateClient, collection_name: str, 
                 source_uuid: str, target_modality: str, limit: int = TOP_K) -> Tuple[List[str], float]:
    """
    Retrieve items of target_modality similar to source item
    Returns: (list of UUIDs, query time in ms)
    """
    collection = client.collections.get(collection_name)
    
    start_time = time.time()
    response = collection.query.near_object(
        near_object=source_uuid,
        limit=limit,
        filters=Filter.by_property("modality").equal(target_modality),
        return_metadata=MetadataQuery(distance=True)
    )
    query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    retrieved_uuids = [str(obj.uuid) for obj in response.objects]
    return retrieved_uuids, query_time


def calculate_precision_at_k(relevant_items: Set[str], retrieved_items: List[str], k: int) -> float:
    """Calculate Precision@K: fraction of retrieved items that are relevant"""
    if k == 0 or not retrieved_items:
        return 0.0
    retrieved_k = retrieved_items[:k]
    relevant_count = sum(1 for item in retrieved_k if item in relevant_items)
    return relevant_count / len(retrieved_k)


def calculate_recall_at_k(relevant_items: Set[str], retrieved_items: List[str], k: int) -> float:
    """Calculate Recall@K: fraction of relevant items that are retrieved"""
    if not relevant_items:
        return 0.0
    if k == 0 or not retrieved_items:
        return 0.0
    retrieved_k = retrieved_items[:k]
    relevant_count = sum(1 for item in retrieved_k if item in relevant_items)
    return relevant_count / len(relevant_items)


def calculate_average_precision(relevant_items: Set[str], retrieved_items: List[str]) -> float:
    """Calculate Average Precision for a single query"""
    if not relevant_items or not retrieved_items:
        return 0.0
    
    score: float = 0.0
    num_relevant: int = 0
    
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            score += precision_at_i
    
    return score / len(relevant_items) if relevant_items else 0.0


def calculate_hit_at_k(relevant_items: Set[str], retrieved_items: List[str], k: int) -> float:
    """Calculate Hit@K: 1 if at least one relevant item in top K, 0 otherwise"""
    if k == 0 or not retrieved_items:
        return 0.0
    retrieved_k = retrieved_items[:k]
    return 1.0 if any(item in relevant_items for item in retrieved_k) else 0.0


def evaluate_retrieval(queries: List[Item], target_items: List[Item], 
                       client: weaviate.WeaviateClient, collection_name: str,
                       target_modality: str, k: int = TOP_K) -> MetricsResult:
    """Evaluate cross-modal retrieval performance"""
    
    # Build mapping from handle to target items
    handle_to_targets: Dict[str, List[str]] = {}
    for item in target_items:
        if item.handle:
            if item.handle not in handle_to_targets:
                handle_to_targets[item.handle] = []
            handle_to_targets[item.handle].append(item.uuid)
    
    precisions: List[float] = []
    recalls: List[float] = []
    avg_precisions: List[float] = []
    hits: List[float] = []
    query_times: List[float] = []
    
    for query in tqdm(queries, desc=f"Evaluating {target_modality}"):
        # Get ground truth relevant items (same handle)
        relevant_handles = {query.handle} if query.handle else set()
        relevant_uuids: Set[str] = set()
        for h in relevant_handles:
            if h in handle_to_targets:
                relevant_uuids.update(handle_to_targets[h])
        
        if not relevant_uuids:
            continue
        
        # Retrieve similar items
        retrieved_uuids, query_time = near_modality(client, collection_name, query.uuid, target_modality, limit=k)
        query_times.append(query_time)
        
        # Calculate metrics
        precisions.append(calculate_precision_at_k(relevant_uuids, retrieved_uuids, k))
        recalls.append(calculate_recall_at_k(relevant_uuids, retrieved_uuids, k))
        avg_precisions.append(calculate_average_precision(relevant_uuids, retrieved_uuids))
        hits.append(calculate_hit_at_k(relevant_uuids, retrieved_uuids, k))
    
    num_queries = len(precisions)
    if num_queries == 0:
        return MetricsResult(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    
    return MetricsResult(
        precision_at_k=sum(precisions) / num_queries,
        recall_at_k=sum(recalls) / num_queries,
        mean_avg_precision=sum(avg_precisions) / num_queries,
        hit_rate_at_k=sum(hits) / num_queries,
        num_queries=num_queries,
        avg_query_time_ms=sum(query_times) / len(query_times) if query_times else 0.0
    )


def print_retrieval_examples(queries: List[Item], client: weaviate.WeaviateClient,
                            collection_name: str, target_modality: str, 
                            k: int = TOP_K, max_examples: int = 3) -> None:
    """Print example retrievals"""
    
    for idx, query in enumerate(queries[:max_examples]):
        retrieved_uuids, _ = near_modality(client, collection_name, query.uuid, target_modality, limit=k)
        
        # Get details of retrieved items
        collection = client.collections.get(collection_name)
        
        header = query.title or query.caption or query.description or ""
        if query.modality == "text":
            print(f"\n[TEXT {idx+1}] {header[:80]}...")
        else:
            print(f"\n[IMAGE {idx+1}] {header[:80]}...")
            print(f"  URL: {query.image_url}")
        
        for i, uuid in enumerate(retrieved_uuids, 1):
            try:
                obj = collection.query.fetch_object_by_id(uuid)
                if obj:
                    props = obj.properties
                    label = props.get("title") or props.get("caption") or ""
                    handle_match = "✓" if props.get("handle") == query.handle else "✗"
                    
                    if target_modality == "image":
                        print(f"  {i}. {handle_match} [image] {label[:60]}")
                    else:
                        print(f"  {i}. {handle_match} [text] {label[:60]}")
            except Exception:
                pass


def benchmark_collection(client: weaviate.WeaviateClient, collection_name: str, k: int = TOP_K) -> Dict:
    """Run complete benchmark on a collection"""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {collection_name}")
    print(f"{'='*80}")
    
    # Load items
    print(f"\nLoading items from collection...")
    items = load_all_items(client, collection_name)
    
    texts = [it for it in items if (it.modality or "").lower() == "text"]
    images = [it for it in items if (it.modality or "").lower() == "image"]
    
    print(f"✓ Loaded {len(texts)} text items and {len(images)} image items")
    
    # Evaluate Text→Image
    print(f"\n{'-'*80}")
    print(f"TEXT → IMAGE RETRIEVAL")
    print(f"{'-'*80}")
    t2i_metrics = evaluate_retrieval(texts, images, client, collection_name, "image", k=k)
    
    print(f"\nResults:")
    print(f"  Number of queries:      {t2i_metrics.num_queries}")
    print(f"  Precision@{k}:           {t2i_metrics.precision_at_k:.4f}")
    print(f"  Recall@{k}:              {t2i_metrics.recall_at_k:.4f}")
    print(f"  Mean Avg Precision:     {t2i_metrics.mean_avg_precision:.4f}")
    print(f"  Hit Rate@{k}:            {t2i_metrics.hit_rate_at_k:.4f}")
    print(f"  Avg Query Time:         {t2i_metrics.avg_query_time_ms:.2f}ms")
    
    # Evaluate Image→Text
    print(f"\n{'-'*80}")
    print(f"IMAGE → TEXT RETRIEVAL")
    print(f"{'-'*80}")
    i2t_metrics = evaluate_retrieval(images, texts, client, collection_name, "text", k=k)
    
    print(f"\nResults:")
    print(f"  Number of queries:      {i2t_metrics.num_queries}")
    print(f"  Precision@{k}:           {i2t_metrics.precision_at_k:.4f}")
    print(f"  Recall@{k}:              {i2t_metrics.recall_at_k:.4f}")
    print(f"  Mean Avg Precision:     {i2t_metrics.mean_avg_precision:.4f}")
    print(f"  Hit Rate@{k}:            {i2t_metrics.hit_rate_at_k:.4f}")
    print(f"  Avg Query Time:         {i2t_metrics.avg_query_time_ms:.2f}ms")
    
    # Show examples
    print(f"\n{'-'*80}")
    print(f"RETRIEVAL EXAMPLES")
    print(f"{'-'*80}")
    print(f"\nText → Image (showing top 3 queries, ✓ = correct handle match)")
    print_retrieval_examples(texts, client, collection_name, "image", k=k, max_examples=3)
    
    print(f"\n\nImage → Text (showing top 3 queries, ✓ = correct handle match)")
    print_retrieval_examples(images, client, collection_name, "text", k=k, max_examples=3)
    
    return {
        "collection": collection_name,
        "num_texts": len(texts),
        "num_images": len(images),
        "text_to_image": {
            "precision_at_k": t2i_metrics.precision_at_k,
            "recall_at_k": t2i_metrics.recall_at_k,
            "mean_avg_precision": t2i_metrics.mean_avg_precision,
            "hit_rate_at_k": t2i_metrics.hit_rate_at_k,
            "num_queries": t2i_metrics.num_queries,
            "avg_query_time_ms": t2i_metrics.avg_query_time_ms
        },
        "image_to_text": {
            "precision_at_k": i2t_metrics.precision_at_k,
            "recall_at_k": i2t_metrics.recall_at_k,
            "mean_avg_precision": i2t_metrics.mean_avg_precision,
            "hit_rate_at_k": i2t_metrics.hit_rate_at_k,
            "num_queries": i2t_metrics.num_queries,
            "avg_query_time_ms": i2t_metrics.avg_query_time_ms
        }
    }


def print_comparison(clip_results: Dict, openai_results: Dict, k: int):
    """Print side-by-side comparison"""
    
    print(f"\n\n{'='*80}")
    print(f"COMPARISON SUMMARY (K={k})")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<30} {'CLIP':>15} {'OpenAI':>15} {'Winner':>15}")
    print(f"{'-'*77}")
    
    # Text→Image metrics
    print(f"\nTEXT → IMAGE:")
    metrics = [
        ("Precision@K", "precision_at_k"),
        ("Recall@K", "recall_at_k"),
        ("Mean Avg Precision", "mean_avg_precision"),
        ("Hit Rate@K", "hit_rate_at_k"),
        ("Avg Query Time (ms)", "avg_query_time_ms")
    ]
    
    for name, key in metrics:
        clip_val = clip_results["text_to_image"][key]
        openai_val = openai_results["text_to_image"][key]
        
        if key == "avg_query_time_ms":
            winner = "CLIP" if clip_val < openai_val else "OpenAI"
            print(f"  {name:<28} {clip_val:>15.2f} {openai_val:>15.2f} {winner:>15}")
        else:
            winner = "CLIP" if clip_val > openai_val else "OpenAI"
            print(f"  {name:<28} {clip_val:>15.4f} {openai_val:>15.4f} {winner:>15}")
    
    # Image→Text metrics
    print(f"\nIMAGE → TEXT:")
    for name, key in metrics:
        clip_val = clip_results["image_to_text"][key]
        openai_val = openai_results["image_to_text"][key]
        
        if key == "avg_query_time_ms":
            winner = "CLIP" if clip_val < openai_val else "OpenAI"
            print(f"  {name:<28} {clip_val:>15.2f} {openai_val:>15.2f} {winner:>15}")
        else:
            winner = "CLIP" if clip_val > openai_val else "OpenAI"
            print(f"  {name:<28} {clip_val:>15.4f} {openai_val:>15.4f} {winner:>15}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CLIP vs OpenAI multimodal retrieval")
    parser.add_argument("--k", type=int, default=5, help="Top-K for evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    print("="*80)
    print("MULTIMODAL RETRIEVAL BENCHMARK")
    print("CLIP vs OpenAI")
    print("="*80)
    
    # Connect to Weaviate
    print(f"\nConnecting to Weaviate...")
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )
    
    try:
        if not client.is_ready():
            print("✗ Weaviate is not ready!")
            sys.exit(1)
        print("✓ Connected")
        
        # Check collections exist
        clip_collection = "MyrityProducts_CLIP"
        openai_collection = "MyrityProducts_OpenAI"
        
        if not client.collections.exists(clip_collection):
            print(f"\n✗ Collection '{clip_collection}' not found!")
            print(f"  Run: python load_clip_collection.py")
            sys.exit(1)
        
        if not client.collections.exists(openai_collection):
            print(f"\n✗ Collection '{openai_collection}' not found!")
            print(f"  Run: python load_openai_collection.py")
            sys.exit(1)
        
        # Benchmark both collections
        clip_results = benchmark_collection(client, clip_collection, k=args.k)
        openai_results = benchmark_collection(client, openai_collection, k=args.k)
        
        # Print comparison
        print_comparison(clip_results, openai_results, args.k)
        
        # Save results
        results = {
            "k": args.k,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clip": clip_results,
            "openai": openai_results
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()

