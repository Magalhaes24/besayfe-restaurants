#!/usr/bin/env python
"""
Training script to process technical sheet examples and train the ML model.
Processes all PDFs in test-technical-sheets/rl/ and learns from them.
"""

import asyncio
import json
from pathlib import Path
from app.config import settings
from app.ml import initialize, get_normalizer
from app.parsers.pdf_parser import extract_pdf


async def train_on_technical_sheets():
    """Process all technical sheet examples and train the model."""

    # Initialize ML system
    settings.ml_dir.mkdir(parents=True, exist_ok=True)
    initialize(settings.ml_dir)

    normalizer = get_normalizer()
    test_dir = Path("test-technical-sheets/rl")
    pdfs = sorted(list(test_dir.glob("*.pdf")))

    results = {
        "processed": [],
        "errors": [],
        "statistics": {
            "total_pdfs": len(pdfs),
            "successful": 0,
            "failed": 0,
            "total_ingredients": 0,
            "total_allergens": 0,
        }
    }

    print(f"\n{'='*70}")
    print("TRAINING ON TECHNICAL SHEET EXAMPLES")
    print(f"{'='*70}")
    print(f"Found {len(pdfs)} PDF files to process\n")

    for i, pdf_path in enumerate(pdfs, 1):
        try:
            print(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.name.encode('utf-8', errors='ignore').decode('utf-8')}")
        except:
            print(f"\n[{i}/{len(pdfs)}] Processing file {i}")
        print("-" * 70)

        try:
            # Extract PDF content - read file as bytes
            with open(pdf_path, 'rb') as f:
                raw_text = await extract_pdf(f.read())

            # Normalize the sheet
            sheet = await normalizer.normalize(
                raw_text,
                source_file=pdf_path.name,
                source_format="pdf"
            )

            # Feed into learning system
            normalizer.rule_engine.learn_from_normalized_sheet(sheet)

            # Record results
            result = {
                "file": pdf_path.name,
                "status": "success",
                "ingredients_count": len(sheet.ingredients),
                "allergens_count": sum(len(ing.allergens) for ing in sheet.ingredients),
                "product_name": sheet.name,
                "category": sheet.category,
            }
            results["processed"].append(result)
            results["statistics"]["successful"] += 1
            results["statistics"]["total_ingredients"] += len(sheet.ingredients)
            results["statistics"]["total_allergens"] += sum(len(ing.allergens) for ing in sheet.ingredients)

            print("[OK] SUCCESS")
            print(f"  Product: {sheet.name}")
            print(f"  Ingredients: {len(sheet.ingredients)}")
            print(f"  Allergens: {sum(len(ing.allergens) for ing in sheet.ingredients)}")

        except Exception as e:
            print(f"[ERROR] {str(e)[:100]}")
            results["errors"].append({
                "file": pdf_path.name,
                "error": str(e)[:200]
            })
            results["statistics"]["failed"] += 1

    # Print summary
    try:
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
    except:
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
    print(f"Processed: {results['statistics']['successful']}/{results['statistics']['total_pdfs']}")
    print(f"Failed: {results['statistics']['failed']}")
    print(f"Total Ingredients Learned: {results['statistics']['total_ingredients']}")
    print(f"Total Allergen-Ingredient Pairs: {results['statistics']['total_allergens']}")

    # Get learning statistics
    learning_stats = normalizer.rule_engine.get_learning_confidence()
    print(f"\nLearning Statistics:")
    print(f"  Rules Learned: {learning_stats.get('total_rules', 0)}")
    print(f"  Avg Rule Confidence: {learning_stats.get('avg_confidence', 0):.2%}")

    # Get semantic graph statistics
    graph_stats = normalizer.rule_engine.semantic_graph.get_graph_statistics()
    print(f"\nSemantic Graph:")
    print(f"  Total Tokens: {graph_stats['total_tokens']}")
    print(f"  Ingredients: {graph_stats['ingredients']}")
    print(f"  Allergens: {graph_stats['allergens']}")
    print(f"  Relationships: {graph_stats['total_relationships']}")
    print(f"  Avg Relationship Strength: {graph_stats['avg_relationship_strength']:.2%}")

    # Save results to file
    results_file = Path("training_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    results = asyncio.run(train_on_technical_sheets())
