#!/usr/bin/env python3
"""
Generate synthetic multimodal dataset for benchmarking
Creates images and associated text descriptions
"""

import os
import csv
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import random

# Categories and their properties
CATEGORIES = {
    "animals": {
        "colors": ["brown", "golden", "white", "black", "gray", "orange"],
        "subjects": ["dog", "cat", "horse", "elephant", "lion", "bird"],
        "actions": ["running", "sitting", "sleeping", "playing", "eating", "jumping"],
        "locations": ["in a field", "in the forest", "at home", "in the zoo", "in nature", "in a park"]
    },
    "vehicles": {
        "colors": ["red", "blue", "black", "white", "silver", "green"],
        "subjects": ["car", "truck", "motorcycle", "bicycle", "bus", "van"],
        "actions": ["parked", "driving", "racing", "stationary", "moving", "speeding"],
        "locations": ["on the road", "in a parking lot", "on the street", "in the city", "on a highway", "in traffic"]
    },
    "nature": {
        "colors": ["green", "blue", "brown", "golden", "red", "purple"],
        "subjects": ["mountain", "ocean", "forest", "sunset", "lake", "waterfall"],
        "actions": ["scenic", "peaceful", "majestic", "beautiful", "serene", "stunning"],
        "locations": ["landscape", "view", "panorama", "vista", "scenery", "environment"]
    },
    "food": {
        "colors": ["golden", "red", "green", "brown", "white", "yellow"],
        "subjects": ["pizza", "burger", "salad", "pasta", "sushi", "sandwich"],
        "actions": ["fresh", "delicious", "hot", "cold", "prepared", "served"],
        "locations": ["on a plate", "on a table", "in a bowl", "in a restaurant", "homemade", "gourmet"]
    },
    "architecture": {
        "colors": ["white", "gray", "brown", "red", "blue", "modern"],
        "subjects": ["building", "house", "skyscraper", "bridge", "tower", "cathedral"],
        "actions": ["tall", "modern", "historic", "beautiful", "impressive", "architectural"],
        "locations": ["in the city", "downtown", "urban", "in Europe", "landmark", "structure"]
    }
}

def create_synthetic_image(text, size=(400, 300), bg_color=None):
    """Create a simple synthetic image with text overlay"""
    
    # Generate background color based on text if not provided
    if bg_color is None:
        random.seed(hash(text) % 1000000)
        bg_color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
    
    # Create image
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw border
    draw.rectangle([(10, 10), (size[0]-10, size[1]-10)], outline="white", width=3)
    
    # Add geometric shapes based on category
    category = text.split()[0].lower() if text else "default"
    random.seed(hash(text) % 1000000)
    
    # Add some random shapes
    for _ in range(3):
        x1, y1 = random.randint(50, size[0]-100), random.randint(50, size[1]-100)
        x2, y2 = x1 + random.randint(30, 80), y1 + random.randint(30, 80)
        shape_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
        
        shape_type = random.choice(['rectangle', 'ellipse'])
        if shape_type == 'rectangle':
            draw.rectangle([(x1, y1), (x2, y2)], fill=shape_color, outline="white")
        else:
            draw.ellipse([(x1, y1), (x2, y2)], fill=shape_color, outline="white")
    
    # Add text in center
    words = text.split()[:5]  # Take first 5 words
    text_short = ' '.join(words)
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text_short, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = (size[0] - text_width) // 2
    text_y = (size[1] - text_height) // 2
    
    # Draw text with shadow for better visibility
    draw.text((text_x + 2, text_y + 2), text_short, fill="black", font=font)
    draw.text((text_x, text_y), text_short, fill="white", font=font)
    
    # Add a label at the top
    label = f"ID: {hash(text) % 10000}"
    draw.text((20, 20), label, fill="white", font=small_font)
    
    return img

def generate_description(category_name):
    """Generate a text description from category templates"""
    category = CATEGORIES[category_name]
    
    color = random.choice(category["colors"])
    subject = random.choice(category["subjects"])
    action = random.choice(category["actions"])
    location = random.choice(category["locations"])
    
    # Generate varied sentence structures
    templates = [
        f"A {color} {subject} {action} {location}",
        f"{color.capitalize()} {subject} {action} {location}",
        f"Beautiful {color} {subject} {action} {location}",
        f"Amazing {subject} in {color} color {action} {location}",
        f"Stunning {color} {subject} {location} {action}",
    ]
    
    return random.choice(templates)

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_dataset(num_samples=50, output_dir="synthetic_data", output_csv="synthetic_dataset.csv"):
    """Generate synthetic dataset with images and descriptions"""
    
    print(f"🎨 Generating synthetic dataset with {num_samples} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = []
    category_list = list(CATEGORIES.keys())
    
    # Generate balanced dataset across categories
    samples_per_category = num_samples // len(category_list)
    remainder = num_samples % len(category_list)
    
    sample_id = 0
    
    for cat_idx, category in enumerate(category_list):
        # Add extra sample to first categories if there's a remainder
        num_cat_samples = samples_per_category + (1 if cat_idx < remainder else 0)
        
        for i in range(num_cat_samples):
            # Generate description
            description = generate_description(category)
            
            # Create image
            img = create_synthetic_image(description)
            
            # Save image
            image_filename = f"image_{sample_id:04d}.png"
            image_path = os.path.join(output_dir, image_filename)
            img.save(image_path)
            
            # Convert to base64 for embedding
            image_base64 = image_to_base64(img)
            
            # Add to dataset
            dataset.append({
                'id': sample_id,
                'category': category,
                'description': description,
                'image_path': image_path,
                'image_base64': image_base64,
                'filename': image_filename
            })
            
            sample_id += 1
            
            if (sample_id) % 10 == 0:
                print(f"  Generated {sample_id}/{num_samples} samples...")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, output_csv)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'description', 'image_path', 'filename'])
        writer.writeheader()
        for item in dataset:
            writer.writerow({
                'id': item['id'],
                'category': item['category'],
                'description': item['description'],
                'image_path': item['image_path'],
                'filename': item['filename']
            })
    
    print(f"✅ Dataset generated successfully!")
    print(f"   Images saved to: {output_dir}/")
    print(f"   CSV saved to: {csv_path}")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Categories: {', '.join(category_list)}")
    
    # Print category distribution
    print("\n📊 Category Distribution:")
    for category in category_list:
        count = sum(1 for item in dataset if item['category'] == category)
        print(f"   {category}: {count} samples")
    
    return dataset, csv_path

def create_ground_truth_queries(dataset, num_queries=20):
    """Create ground truth queries for evaluation"""
    
    print(f"\n🎯 Generating {num_queries} ground truth queries...")
    
    queries = []
    
    # Create queries for each category
    for category in CATEGORIES.keys():
        cat_items = [item for item in dataset if item['category'] == category]
        
        if len(cat_items) >= 2:
            # Select a random item as query
            query_item = random.choice(cat_items)
            
            # Ground truth: all items from the same category
            relevant_ids = [item['id'] for item in cat_items if item['id'] != query_item['id']]
            
            queries.append({
                'query_id': query_item['id'],
                'query_text': query_item['description'],
                'query_category': category,
                'relevant_ids': relevant_ids,
                'total_relevant': len(relevant_ids)
            })
    
    # Add some cross-category queries (harder queries)
    query_templates = [
        ("animal", "A pet animal"),
        ("vehicle", "Transportation on the road"),
        ("nature", "Beautiful outdoor scenery"),
        ("food", "Delicious meal"),
        ("architecture", "Urban structure")
    ]
    
    for category, query_text in query_templates[:5]:
        cat_items = [item for item in dataset if item['category'] == category]
        relevant_ids = [item['id'] for item in cat_items]
        
        queries.append({
            'query_id': f"generic_{category}",
            'query_text': query_text,
            'query_category': category,
            'relevant_ids': relevant_ids,
            'total_relevant': len(relevant_ids)
        })
    
    print(f"✅ Generated {len(queries)} ground truth queries")
    
    return queries

if __name__ == "__main__":
    # Generate dataset
    dataset, csv_path = generate_dataset(num_samples=50)
    
    # Generate ground truth queries
    queries = create_ground_truth_queries(dataset)
    
    # Save queries
    queries_path = os.path.join("synthetic_data", "ground_truth_queries.csv")
    with open(queries_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'query_text', 'query_category', 'relevant_ids', 'total_relevant'])
        writer.writeheader()
        for query in queries:
            writer.writerow({
                'query_id': query['query_id'],
                'query_text': query['query_text'],
                'query_category': query['query_category'],
                'relevant_ids': ','.join(map(str, query['relevant_ids'])),
                'total_relevant': query['total_relevant']
            })
    
    print(f"\n✅ Ground truth queries saved to: {queries_path}")
    print("\n🎉 Dataset generation complete!")
    print(f"\n📝 Usage:")
    print(f"   - Dataset CSV: synthetic_data/synthetic_dataset.csv")
    print(f"   - Ground truth: synthetic_data/ground_truth_queries.csv")
    print(f"   - Images: synthetic_data/image_*.png")


