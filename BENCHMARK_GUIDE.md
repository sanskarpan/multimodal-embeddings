# Multimodal Embedding Benchmark Guide

## Overview

This guide explains how to benchmark multimodal embeddings using the provided tools. The benchmark compares **CLIP** (multi2vec-clip) and **OpenAI** (text-embedding-3-small) on multimodal retrieval tasks.

---

## Quick Start

### 1. Generate Synthetic Dataset
```bash
source venv/bin/activate
python generate_synthetic_dataset.py
```

This creates:
- `synthetic_data/synthetic_dataset.csv` - Dataset metadata
- `synthetic_data/ground_truth_queries.csv` - Evaluation queries
- `synthetic_data/image_*.png` - 50 synthetic images

### 2. Run Benchmark
```bash
python benchmark_multimodal.py
```

This will:
- Create CLIP and OpenAI collections
- Insert data into both collections
- Run evaluation queries
- Calculate metrics (Precision@K, Recall@K, mAP, Hit Rate@K)
- Save results to `benchmark_results.json`

### 3. View Results
```bash
cat benchmark_results.json
```

Or read the comprehensive report:
```bash
cat BENCHMARK_REPORT.md
```

---

## Using Your Own Dataset

### CSV Format

Your dataset CSV should have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int | Unique identifier | 0, 1, 2, ... |
| `description` | text | Text description | "A golden dog playing" |
| `category` | text | Category label | "animals" |
| `image_path` | text | Path to image | "images/dog_001.png" |
| `filename` | text | Image filename | "dog_001.png" |

### Example CSV
```csv
id,description,category,image_path,filename
0,A golden dog playing in a park,animals,images/dog_001.png,dog_001.png
1,Red car parked on the street,vehicles,images/car_001.png,car_001.png
2,Beautiful mountain landscape,nature,images/mountain_001.png,mountain_001.png
```

### Ground Truth Queries Format

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `query_id` | text/int | Query identifier | "query_1" |
| `query_text` | text | Query string | "pet animals" |
| `query_category` | text | Expected category | "animals" |
| `relevant_ids` | text | Comma-separated IDs | "0,5,12,18" |
| `total_relevant` | int | Number of relevant items | 4 |

### Example Ground Truth CSV
```csv
query_id,query_text,query_category,relevant_ids,total_relevant
query_1,A pet animal,animals,"0,5,12,18",4
query_2,Transportation on the road,vehicles,"1,7,15,22",4
```

### Modify Benchmark Script

Edit `benchmark_multimodal.py`:

```python
# Load your custom dataset
dataset_csv = "path/to/your_dataset.csv"
queries_csv = "path/to/your_queries.csv"

benchmark.load_dataset(dataset_csv)
benchmark.load_ground_truth(queries_csv)
```

---

## Configuration

### Adjust K Values

In `benchmark_multimodal.py`, modify:

```python
# Default K values

# Custom K values
k_values = [1, 3, 5, 10]
k_values = [1, 5, 10, 20, 50]
```

### Change Collection Names

```python
# CLIP collection
benchmark.create_collection_clip("MyCustomCLIP")
benchmark.insert_data_clip("MyCustomCLIP")

# OpenAI collection
benchmark.create_collection_openai("MyCustomOpenAI")
benchmark.insert_data_openai("MyCustomOpenAI")
```

### Adjust Wait Times

After inserting data, the script waits for indexing:

```python
# Default: 3 seconds
time.sleep(3)

# For larger datasets, increase wait time
time.sleep(10)
```

---

## Metrics Explained

### Precision@K
**What**: Fraction of retrieved items (top-K) that are relevant  
**Formula**: `True Positives / K`  
**Range**: 0.0 to 1.0 (higher is better)  
**Example**: If you retrieve 10 items and 7 are relevant, P@10 = 0.7

### Recall@K
**What**: Fraction of all relevant items that appear in top-K  
**Formula**: `True Positives / Total Relevant`  
**Range**: 0.0 to 1.0 (higher is better)  
**Example**: If there are 20 relevant items and you retrieve 15 of them, R@20 = 0.75

### Hit Rate@K
**What**: Percentage of queries with at least one relevant result in top-K  
**Formula**: `(Queries with ≥1 relevant) / Total Queries`  
**Range**: 0.0 to 1.0 (higher is better)  
**Example**: If 8 out of 10 queries have a relevant result, HR@K = 0.8

### Mean Average Precision (mAP)
**What**: Average precision across all queries, considering ranking  
**Formula**: Average of AP for each query  
**Range**: 0.0 to 1.0 (higher is better)  
**Why Important**: Rewards models that rank relevant items higher

---

## Synthetic Dataset Generation

### Customize Categories

Edit `generate_synthetic_dataset.py`:

```python
CATEGORIES = {
    "category_name": {
        "colors": ["red", "blue", "green"],
        "subjects": ["item1", "item2"],
        "actions": ["action1", "action2"],
        "locations": ["place1", "place2"]
    },
    # Add more categories...
}
```

### Change Dataset Size

```python
# Generate 100 samples instead of 50
dataset, csv_path = generate_dataset(num_samples=100)
```

### Customize Image Appearance

```python
# Change image size
img = create_synthetic_image(text, size=(800, 600))

# Change colors
bg_color = (255, 200, 150)  # Custom background color
```

---

## 🔍 Advanced Usage

### Run Single Model Benchmark

To benchmark only CLIP:

```python
# In benchmark_multimodal.py
# Comment out OpenAI sections:

# benchmark.create_collection_openai("MultimodalOpenAI")
# benchmark.insert_data_openai("MultimodalOpenAI")
# openai_metrics, openai_results = benchmark.search_and_evaluate(...)
```

### Export Results to CSV

Add to `benchmark_multimodal.py`:

```python
import csv

def export_to_csv(metrics, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'K', 'Value'])
        
        for k, v in metrics['precision_at_k'].items():
            writer.writerow(['Precision', k, v])
        
        for k, v in metrics['recall_at_k'].items():
            writer.writerow(['Recall', k, v])
        
        writer.writerow(['mAP', '-', metrics['map']])

# Use it:
export_to_csv(clip_metrics, 'clip_results.csv')
export_to_csv(openai_metrics, 'openai_results.csv')
```

### Visualize Results

```python
import matplotlib.pyplot as plt

def plot_comparison(clip_metrics, openai_metrics):
    k_values = list(clip_metrics['precision_at_k'].keys())
    
    clip_precision = [clip_metrics['precision_at_k'][k] for k in k_values]
    openai_precision = [openai_metrics['precision_at_k'][k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, clip_precision, 'o-', label='CLIP', linewidth=2)
    plt.plot(k_values, openai_precision, 's-', label='OpenAI', linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Precision@K')
    plt.title('Precision@K Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_comparison.png')
    plt.show()
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution**: Run `python generate_synthetic_dataset.py` first

### Issue: "Connection refused"
**Solution**: Start Weaviate with `./setup_weaviate.sh start`

### Issue: "OpenAI API error"
**Solution**: Check your API key in `.env` file and verify you have credits

### Issue: "CLIP embeddings fail"
**Solution**: 
- Check multi2vec-clip container is running: `docker ps`
- View logs: `docker logs multi2vec-clip`
- Restart: `./setup_weaviate.sh restart`

### Issue: "Slow OpenAI queries"
**Solution**: This is normal - OpenAI requires API calls (~1 second each)

### Issue: "Out of memory"
**Solution**: 
- Reduce dataset size
- Process in smaller batches
- Check Docker memory limits

---

## Sample Results

### Expected Performance (50-item dataset)

**CLIP:**
- Precision@10: ~0.50-0.60
- Recall@10: ~0.50-0.60
- mAP: ~0.65-0.75
- Query Time: 20-30ms

**OpenAI:**
- Precision@10: ~0.70-0.80
- Recall@10: ~0.70-0.80
- mAP: ~0.80-0.90
- Query Time: 800-1200ms

---

## Best Practices

### 1. Dataset Quality
- Use high-quality, diverse images
- Write clear, descriptive text
- Balance categories evenly
- Include challenging negative examples

### 2. Ground Truth
- Create meaningful queries
- Ensure fair category representation
- Include both easy and hard queries
- Validate relevance manually

### 3. Evaluation
- Test multiple K values
- Run multiple times for stability
- Use statistical significance tests
- Consider domain-specific metrics

### 4. Optimization
- Cache embeddings when possible
- Batch process for efficiency
- Monitor API costs (OpenAI)
- Profile performance bottlenecks

---

## Additional Resources

### Code Files
- `generate_synthetic_dataset.py` - Dataset generator
- `benchmark_multimodal.py` - Benchmark runner
- `BENCHMARK_REPORT.md` - Detailed results report

### Documentation
- `README.md` - Weaviate setup guide
- `QUICK_START.md` - Quick reference
- `SETUP_COMPLETE.md` - Setup summary

### External Links
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

---

## Contributing

To add new metrics or features:

1. **Add Metric Calculation**: In `benchmark_multimodal.py`
```python
def calculate_new_metric(results):
    # Your metric calculation
    return value
```

2. **Update Display**: In `print_metrics()`
```python
print(f"New Metric: {metrics['new_metric']:.4f}")
```

3. **Save to Results**: In `save_results()`
```python
results['clip']['metrics']['new_metric'] = new_value
```

---

**Version**: 1.0  
**Python**: 3.12+  
**Weaviate**: 1.27.3


