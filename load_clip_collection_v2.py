#!/usr/bin/env python3
"""
Load Myriti products into Weaviate using CLIP with ACTUAL IMAGES
Uses downloaded images converted to base64 for proper multimodal embeddings
"""

import os
import sys
import base64
from pathlib import Path
from typing import List, Optional, Tuple
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from dotenv import load_dotenv
from tqdm import tqdm

from data_loader import Product, parse_myriti_csv, get_product_text, get_product_caption
from download_images import download_product_images

load_dotenv()

COLLECTION_NAME = "MyrityProducts_CLIP_Real"
CSV_PATH = "reference/myriti_products.csv"
IMAGE_CACHE_DIR = "image_cache"


def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"  Error encoding {image_path}: {e}")
        return None


def recreate_collection(client: weaviate.WeaviateClient) -> None:
    """Create or recreate the CLIP-based collection"""
    print(f"Creating collection '{COLLECTION_NAME}' with CLIP multimodal embeddings...")
    
    # Delete if exists
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            print(f"  Deleted existing collection")
    except Exception as e:
        print(f"  Warning: {e}")
    
    # Create collection with CLIP vectorizer
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.multi2vec_clip(
            text_fields=["title", "description", "caption"],
            image_fields=["image"]  # This BLOB field will contain actual images
        ),
        properties=[
            Property(name="handle", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="caption", data_type=DataType.TEXT),
            Property(name="image", data_type=DataType.BLOB),  # Actual image data
            Property(name="image_url", data_type=DataType.TEXT),
            Property(name="modality", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="vendor", data_type=DataType.TEXT),
            Property(name="product_type", data_type=DataType.TEXT),
        ]
    )
    print(f"  ✓ Collection created")


def insert_products_with_images(client: weaviate.WeaviateClient, 
                                products: List[Product],
                                image_map: dict) -> int:
    """
    Insert products with BOTH text AND images in SINGLE objects
    CLIP will create unified multimodal embeddings from both
    """
    collection = client.collections.get(COLLECTION_NAME)
    
    inserted = 0
    failed = 0
    
    print(f"\nInserting products with MULTIMODAL embeddings (text + image)...")
    print(f"   CLIP will process BOTH text and image for each product")
    
    for product in tqdm(products, desc="Products"):
        if product.handle not in image_map:
            failed += 1
            continue
        
        # Ensure we have text
        text = get_product_text(product)
        if not text:
            failed += 1
            continue
        
        try:
            # Load and encode image
            image_path = image_map[product.handle]
            image_b64 = image_to_base64(image_path)
            
            if not image_b64:
                failed += 1
                continue
            
            caption = get_product_caption(product)
            
            # Create SINGLE object with BOTH text AND image
            # CLIP will fuse both modalities into one embedding
            props = {
                "handle": product.handle,
                "title": product.title or "",
                "description": product.description or "",
                "caption": caption,
                "image": image_b64,  # ACTUAL IMAGE DATA
                "image_url": product.image_urls[0] if product.image_urls else "",
                "modality": "multimodal",  # Both text and image
                "tags": product.tags,
                "vendor": product.vendor,
                "product_type": product.product_type,
            }
            
            collection.data.insert(properties=props)
            inserted += 1
            
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Error inserting product {product.handle}: {e}")
    
    print(f"\n✓ Multimodal objects inserted: {inserted} (failed: {failed})")
    print(f"   Each object contains BOTH text and image for unified CLIP embedding")
    
    return inserted


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Myriti products with ACTUAL images")
    parser.add_argument("--sample", type=int, default=None,
                       help="Only process first N products (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip image download (use cached images)")
    args = parser.parse_args()
    
    print("="*80)
    print("CLIP COLLECTION LOADER (WITH REAL IMAGES)")
    print("="*80)
    
    # Parse CSV
    print(f"\nLoading products from {CSV_PATH}...")
    products = parse_myriti_csv(CSV_PATH)
    
    if args.sample:
        products = products[:args.sample]
        print(f"✓ Using sample of {len(products)} products")
    else:
        print(f"✓ Loaded {len(products)} products")
    
    # Download images
    if not args.skip_download:
        image_map = download_product_images(products)
    else:
        # Use cached images
        print(f"\nUsing cached images from {IMAGE_CACHE_DIR}/...")
        image_cache = Path(IMAGE_CACHE_DIR)
        image_map = {}
        for product in products:
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                path = image_cache / f"{product.handle}.{ext}"
                if path.exists():
                    image_map[product.handle] = str(path)
                    break
        print(f"✓ Found {len(image_map)} cached images")
    
    if len(image_map) == 0:
        print("\n✗ No images available! Run without --skip-download first")
        sys.exit(1)
    
    # Connect to Weaviate
    print(f"\nConnecting to Weaviate...")
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )
    
    try:
        if not client.is_ready():
            print("✗ Weaviate is not ready!")
            print("  Please start it with: ./setup_weaviate.sh start")
            sys.exit(1)
        print("✓ Connected to Weaviate")
        
        # Create collection
        recreate_collection(client)
        
        # Filter products to only those with images
        products_with_images = [p for p in products if p.handle in image_map]
        print(f"\n✓ Processing {len(products_with_images)} products with images")
        
        # Insert products
        count = insert_products_with_images(
            client, products_with_images, image_map
        )
        
        # Verify
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.aggregate.over_all(total_count=True)
        total = response.total_count
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total objects in collection: {total}")
        print(f"  - Multimodal objects:  {count} (EACH with TEXT + IMAGE)")
        print(f"\n✓ CLIP collection ready with unified multimodal embeddings!")
        
        if args.sample:
            cost = len(products_with_images) * 0.0  # CLIP is free
            print(f"\nEstimated cost: $0 (CLIP is local/free)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()

