#!/usr/bin/env python3
"""
Manually insert the failed product: orange-paithani-silk-lehenga
"""

import os
import sys
import base64
from pathlib import Path
from typing import Optional
import weaviate
from dotenv import load_dotenv
from openai import OpenAI

from data_loader import parse_myriti_csv, get_product_text, get_product_caption

load_dotenv()


def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None


def generate_image_description_from_file(image_path: str, openai_client: OpenAI, product_title: str) -> Optional[str]:
    """Generate description from local image file using GPT-4o-mini vision"""
    try:
        image_b64 = image_to_base64(image_path)
        if not image_b64:
            return None
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe this product image in detail. Product: {product_title}. Focus on colors, patterns, style, fabric, embellishments, and overall appearance."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating description: {e}")
        return None


def main():
    print("="*80)
    print("MANUALLY INSERTING FAILED PRODUCT")
    print("="*80)
    
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    print("\n✓ Connected to Weaviate")

    collection = client.collections.get("MyrityProducts_OpenAI_Real")

    # Parse CSV to find the failed product
    print("\nLoading products...")
    products = parse_myriti_csv("reference/myriti_products.csv")
    failed_product = None

    for p in products:
        if p.handle == "orange-paithani-silk-lehenga":
            failed_product = p
            break

    if not failed_product:
        print("✗ Failed product not found in CSV")
        sys.exit(1)

    print(f"\n✓ Found failed product:")
    print(f"  Handle: {failed_product.handle}")
    print(f"  Title: {failed_product.title}")

    # Check if image exists
    image_path = f"image_cache/{failed_product.handle}.jpg"
    if not os.path.exists(image_path):
        print(f"\n✗ Image not found: {image_path}")
        sys.exit(1)

    print(f"  ✓ Image found: {image_path}")

    # Generate description
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("\nGenerating AI vision description...")
    image_desc = generate_image_description_from_file(image_path, openai_client, failed_product.title)

    if not image_desc:
        print("  ⚠️  Failed to generate description, using caption fallback")
        image_desc = get_product_caption(failed_product)
    else:
        print(f"  ✓ Generated: {image_desc[:80]}...")

    caption = get_product_caption(failed_product)

    # Insert
    props = {
        "handle": failed_product.handle,
        "title": failed_product.title or "",
        "description": failed_product.description or "",
        "caption": caption,
        "image_description": image_desc or caption,
        "image_url": failed_product.image_urls[0] if failed_product.image_urls else "",
        "modality": "multimodal",
        "tags": failed_product.tags,
        "vendor": failed_product.vendor,
        "product_type": failed_product.product_type,
    }

    print("\nInserting product into collection...")
    try:
        uuid = collection.data.insert(properties=props)
        print(f"✓ Successfully inserted with UUID: {uuid}")
        
        # Verify
        response = collection.aggregate.over_all(total_count=True)
        print(f"\n✓ Total objects in collection: {response.total_count}")
        
        print("\n" + "="*80)
        print("SUCCESS! Product manually inserted.")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Failed to insert: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    client.close()


if __name__ == "__main__":
    main()

