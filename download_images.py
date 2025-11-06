#!/usr/bin/env python3
"""
Download images from Myriti products CSV
Creates a local cache to avoid re-downloading
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import requests
from tqdm import tqdm
from data_loader import parse_myriti_csv, Product

IMAGE_CACHE_DIR = "image_cache"
TIMEOUT = 10.0
MAX_SIZE_MB = 10


def download_image(url: str, save_path: Path) -> bool:
    """Download a single image"""
    try:
        response = requests.get(url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check size
        size = int(response.headers.get('Content-Length', 0))
        if size > MAX_SIZE_MB * 1024 * 1024:
            print(f"  Skipping {url}: too large ({size / 1024 / 1024:.1f}MB)")
            return False
        
        # Download
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
        
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_product_images(products: List[Product], cache_dir: str = IMAGE_CACHE_DIR) -> dict:
    """
    Download all product images
    Returns: {handle: local_image_path}
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    image_map = {}
    downloaded = 0
    cached = 0
    failed = 0
    
    print(f"\nDownloading images to {cache_dir}/...")
    
    for product in tqdm(products, desc="Downloading"):
        if not product.image_urls:
            failed += 1
            continue
        
        # Use first image
        url = product.image_urls[0]
        
        # Create filename from handle
        ext = url.split('.')[-1].split('?')[0]
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        
        filename = f"{product.handle}.{ext}"
        save_path = cache_path / filename
        
        # Check if already downloaded
        if save_path.exists():
            image_map[product.handle] = str(save_path)
            cached += 1
            continue
        
        # Download
        if download_image(url, save_path):
            image_map[product.handle] = str(save_path)
            downloaded += 1
            time.sleep(0.1)  # Be nice to CDN
        else:
            failed += 1
    
    print(f"\n✓ Downloaded: {downloaded}")
    print(f"✓ Cached: {cached}")
    print(f"✗ Failed: {failed}")
    print(f"Total available: {len(image_map)}")
    
    return image_map


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, help="Only download first N products")
    args = parser.parse_args()
    
    print("="*80)
    print("IMAGE DOWNLOADER")
    print("="*80)
    
    # Load products
    products = parse_myriti_csv("reference/myriti_products.csv")
    
    if args.sample:
        products = products[:args.sample]
        print(f"\nUsing sample of {len(products)} products")
    
    print(f"Total products to process: {len(products)}")
    
    # Download
    image_map = download_product_images(products)
    
    print(f"\n✓ Images cached in {IMAGE_CACHE_DIR}/")


if __name__ == "__main__":
    main()

