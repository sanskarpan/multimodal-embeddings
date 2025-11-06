#!/usr/bin/env python3
"""
Data loader for Myriti products CSV
Parses and structures product data for vectorization
"""

import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class Product:
    """Product data structure"""
    handle: str
    title: str
    description: str
    tags: List[str]
    image_urls: List[str]  # Multiple images per product
    vendor: str
    product_type: str


def html_to_text(html: str) -> str:
    """Convert HTML to plain text"""
    if not html:
        return ""
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        # Fallback to regex if BeautifulSoup fails
        text = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', html)
        text = re.sub(r'(?is)<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def parse_myriti_csv(path: str) -> List[Product]:
    """
    Parse Myriti products CSV file
    Groups variants by handle and collects all images
    """
    by_handle: Dict[str, Dict] = {}
    
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            handle = (row.get('Handle') or '').strip()
            if not handle:
                continue
            
            title = (row.get('Title') or '').strip()
            desc_html = row.get('Body (HTML)') or ''
            desc = html_to_text(desc_html)
            vendor = (row.get('Vendor') or '').strip()
            product_type = (row.get('Type') or '').strip()
            
            # Parse tags
            tags_str = (row.get('Tags') or '').strip()
            tags = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else []
            
            # Get image URL
            img = (row.get('Image Src') or '').strip()
            
            # Initialize or update product data
            if handle not in by_handle:
                by_handle[handle] = {
                    'handle': handle,
                    'title': title,
                    'description': desc,
                    'vendor': vendor,
                    'product_type': product_type,
                    'tags': tags,
                    'images': []
                }
            
            # Add image if present and not already added
            if img and img not in by_handle[handle]['images']:
                by_handle[handle]['images'].append(img)
            
            # Update fields if they were empty
            if not by_handle[handle]['title'] and title:
                by_handle[handle]['title'] = title
            if not by_handle[handle]['description'] and desc:
                by_handle[handle]['description'] = desc
            if not by_handle[handle]['tags'] and tags:
                by_handle[handle]['tags'] = tags
    
    # Convert to Product objects
    products = []
    for handle, data in by_handle.items():
        # Only include products with title or description or images
        if data['title'] or data['description'] or data['images']:
            products.append(Product(
                handle=data['handle'],
                title=data['title'],
                description=data['description'],
                tags=data['tags'],
                image_urls=data['images'],
                vendor=data['vendor'],
                product_type=data['product_type']
            ))
    
    return products


def get_product_text(product: Product) -> str:
    """Get combined text representation of product"""
    parts = []
    if product.title:
        parts.append(product.title)
    if product.description:
        # Truncate long descriptions
        desc = product.description[:500] if len(product.description) > 500 else product.description
        parts.append(desc)
    return ' '.join(parts)


def get_product_caption(product: Product) -> str:
    """Get short caption for product"""
    if product.title:
        if product.description and len(product.description) > 0:
            desc_preview = product.description[:100] + '...' if len(product.description) > 100 else product.description
            return f"{product.title}: {desc_preview}"
        return product.title
    elif product.description:
        return product.description[:100] + '...' if len(product.description) > 100 else product.description
    return ""


if __name__ == "__main__":
    # Test the loader
    import sys
    
    csv_path = "reference/myriti_products.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"Loading products from {csv_path}...")
    products = parse_myriti_csv(csv_path)
    
    print(f"\nTotal products: {len(products)}")
    print(f"Products with images: {sum(1 for p in products if p.image_urls)}")
    print(f"Products with text: {sum(1 for p in products if p.title or p.description)}")
    print(f"Total images: {sum(len(p.image_urls) for p in products)}")
    
    # Show sample products
    print("\n" + "="*80)
    print("SAMPLE PRODUCTS:")
    print("="*80)
    for i, p in enumerate(products[:3], 1):
        print(f"\n{i}. {p.handle}")
        print(f"   Title: {p.title}")
        print(f"   Description: {p.description[:100]}..." if len(p.description) > 100 else f"   Description: {p.description}")
        print(f"   Images: {len(p.image_urls)}")
        print(f"   Tags: {', '.join(p.tags[:5])}")

