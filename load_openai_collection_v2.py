#!/usr/bin/env python3
"""
Load Myriti products into Weaviate using OpenAI with ACTUAL image analysis
Uses GPT-4o-mini vision to generate rich image descriptions
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
from openai import OpenAI

from data_loader import Product, parse_myriti_csv, get_product_text, get_product_caption
from download_images import download_product_images

load_dotenv()

COLLECTION_NAME = "MyrityProducts_OpenAI_Real"
CSV_PATH = "reference/myriti_products.csv"
IMAGE_CACHE_DIR = "image_cache"


def recreate_collection(client: weaviate.WeaviateClient) -> None:
    """Create or recreate the OpenAI-based collection"""
    print(f"Creating collection '{COLLECTION_NAME}' with OpenAI embeddings...")
    
    # Delete if exists
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            print(f"  Deleted existing collection")
    except Exception as e:
        print(f"  Warning: {e}")
    
    # Create collection with OpenAI vectorizer
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small",
            vectorize_collection_name=False
        ),
        properties=[
            Property(name="handle", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="caption", data_type=DataType.TEXT),
            Property(name="image_description", data_type=DataType.TEXT),  # AI-generated from actual image
            Property(name="image_url", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="modality", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="tags", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
            Property(name="vendor", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="product_type", data_type=DataType.TEXT, skip_vectorization=True),
        ]
    )
    print(f"  ✓ Collection created")


def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image to base64 for OpenAI API"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return None


def generate_image_description_from_file(image_path: str, openai_client: OpenAI, 
                                         title: str = "") -> Optional[str]:
    """
    Generate description from ACTUAL image file using GPT-4o-mini vision
    Cost: ~$0.0002 per image
    """
    try:
        # Encode image
        base64_image = encode_image_base64(image_path)
        if not base64_image:
            return None
        
        # Determine image type
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        prompt = "Describe this fashion/clothing product image in detail. Focus on: colors, patterns, style, fabric appearance, design elements, and any visible embellishments or decorations."
        if title:
            prompt += f" The product is called: {title}"
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "low"  # Low detail for cost efficiency
                            }
                        }
                    ]
                }
            ],
            max_tokens=150  # Keep descriptions concise
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"  Error generating description for {image_path}: {e}")
        return None


def insert_products_with_image_analysis(client: weaviate.WeaviateClient, 
                                        products: List[Product],
                                        image_map: dict,
                                        use_vision: bool = True) -> Tuple[int, float]:
    """
    Insert products with text AND AI-generated image descriptions in SINGLE objects
    Returns: (count, estimated_cost)
    """
    collection = client.collections.get(COLLECTION_NAME)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    inserted = 0
    failed = 0
    total_cost = 0.0
    
    print(f"\nInserting products with text + AI vision analysis...")
    if use_vision:
        print(f"   Using GPT-4o-mini to analyze images")
        print(f"   Estimated cost: ${len(image_map) * 0.0002:.3f}")
    else:
        print(f"   Skipping vision analysis (text only)")
    
    for product in tqdm(products, desc="Products"):
        if product.handle not in image_map:
            failed += 1
            continue
        
        text = get_product_text(product)
        if not text:
            failed += 1
            continue
        
        try:
            caption = get_product_caption(product)
            image_desc = ""
            
            # Generate image description from actual image
            if use_vision:
                image_path = image_map[product.handle]
                image_desc = generate_image_description_from_file(
                    image_path, openai_client, product.title
                )
                
                if image_desc:
                    total_cost += 0.0002  # Approximate cost per image
            
            # Create SINGLE object with text + image description
            # OpenAI will embed the combined multimodal information
            props = {
                "handle": product.handle,
                "title": product.title or "",
                "description": product.description or "",
                "caption": caption,
                "image_description": image_desc or caption,  # Fallback to caption
                "image_url": product.image_urls[0] if product.image_urls else "",
                "modality": "multimodal",  # Text + vision analysis
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
    
    return inserted, total_cost


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Myriti products with OpenAI vision")
    parser.add_argument("--sample", type=int, default=None,
                       help="Only process first N products (RECOMMENDED for testing)")
    parser.add_argument("--no-vision", action="store_true",
                       help="Skip AI vision analysis (text-only, cheaper)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip image download (use cached images)")
    args = parser.parse_args()
    
    print("="*80)
    print("OPENAI COLLECTION LOADER (WITH REAL IMAGE ANALYSIS)")
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
    
    # Cost warning
    if not args.no_vision:
        estimated_cost = len(image_map) * 0.0002
        print(f"\n⚠️  AI vision analysis enabled")
        print(f"   Estimated cost: ${estimated_cost:.3f} for {len(image_map)} images")
        
        if not args.sample and len(image_map) > 100:
            print(f"\n⚠️  Processing {len(image_map)} images will cost ~${estimated_cost:.2f}")
            print(f"   Consider using --sample 50 for testing first")
            response = input("\n   Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    
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
            print("  Please start it with: ./setup_weaviate.sh start")
            sys.exit(1)
        print("✓ Connected to Weaviate")
        
        # Create collection
        recreate_collection(client)
        
        # Filter products to only those with images
        products_with_images = [p for p in products if p.handle in image_map]
        print(f"\n✓ Processing {len(products_with_images)} products with images")
        
        # Insert products
        count, actual_cost = insert_products_with_image_analysis(
            client, products_with_images, image_map, use_vision=not args.no_vision
        )
        
        # Verify
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.aggregate.over_all(total_count=True)
        total = response.total_count
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total objects in collection: {total}")
        print(f"  - Multimodal objects:  {count} (TEXT + AI VISION)")
        print(f"\nActual cost: ${actual_cost:.3f}")
        print(f"\n✓ OpenAI collection ready with unified multimodal embeddings!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()

