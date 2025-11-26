"""
Test script for enhanced journalist/writer name extraction
Demonstrates the new block-based, NER-enhanced extraction pipeline
"""

import os
import sys
from pathlib import Path
from writer_manager import WriterManager

def test_basic_extraction():
    """Test basic text-based extraction"""
    print("\n" + "="*60)
    print("TEST 1: Basic Text Extraction")
    print("="*60)

    wm = WriterManager()

    sample_text = """
    Breaking News: New Climate Policy Announced

    By Sarah Johnson and Michael Chen

    WASHINGTON - The government today announced a major new climate policy.
    Dr. Emily Roberts from MIT praised the initiative.
    Professor John Williams at Stanford also commented on the development.
    """

    writers = wm.extract_writer_names(sample_text)
    print(f"Found {len(writers)} writers: {writers}")
    return writers


def test_pdf_extraction(pdf_path: str = None):
    """Test enhanced PDF extraction"""
    print("\n" + "="*60)
    print("TEST 2: Enhanced PDF Extraction")
    print("="*60)

    if not pdf_path:
        print("No PDF path provided. Skipping PDF test.")
        print("Usage: python test_enhanced_extraction.py /path/to/test.pdf")
        return []

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return []

    wm = WriterManager()

    print(f"\nProcessing PDF: {pdf_path}")
    print("-" * 60)

    # Step 1: Extract blocks
    print("\nStep 1: Extracting text blocks...")
    blocks = wm.extract_blocks_from_pdf(pdf_path)
    print(f"  → Extracted {len(blocks)} text blocks")

    if blocks:
        # Show sample block
        print(f"\n  Sample block:")
        print(f"    Text: {blocks[0]['text'][:100]}...")
        print(f"    Font size: {blocks[0].get('font_size', 0):.2f}")
        print(f"    Page: {blocks[0].get('page', 0)}")

    # Step 2: Identify headlines
    print("\nStep 2: Identifying headlines...")
    headlines = wm.identify_headline_blocks(blocks)
    print(f"  → Found {len(headlines)} headline blocks")

    if headlines:
        print(f"\n  Sample headline:")
        print(f"    Text: {headlines[0]['text'][:100]}")
        print(f"    Font size: {headlines[0].get('font_size', 0):.2f}")

    # Step 3: Extract bylines
    print("\nStep 3: Extracting byline patterns...")
    byline_writers = wm.extract_byline_patterns(blocks)
    print(f"  → Byline extraction found: {byline_writers}")

    # Step 4: NER extraction
    print("\nStep 4: Running NER on headline-adjacent blocks...")
    ner_writers = wm.extract_names_with_ner(blocks, near_headlines_only=True)
    print(f"  → NER extraction found: {ner_writers}")

    # Step 5: Complete extraction
    print("\nStep 5: Complete extraction pipeline...")
    all_writers = wm.extract_writer_names_from_pdf(pdf_path)
    print(f"  → Total writers extracted: {all_writers}")

    print("\n" + "="*60)
    print(f"FINAL RESULT: {len(all_writers)} unique writers found")
    print("="*60)

    return all_writers


def test_process_document():
    """Test the full document processing workflow"""
    print("\n" + "="*60)
    print("TEST 3: Full Document Processing Workflow")
    print("="*60)

    wm = WriterManager()

    # Simulate processing a document
    sample_text = """
    Special Report: Technology in Education

    By Dr. Jennifer Martinez

    SAN FRANCISCO - A new study by Prof. Robert Chen at Berkeley
    shows promising results. The research, co-authored with
    Dr. Lisa Thompson, reveals significant improvements.
    """

    print("\nProcessing document with sample text...")
    writer_ids = wm.process_document_for_writers(
        document_file="test_article.txt",
        text_content=sample_text
    )

    print(f"\nExtracted and stored {len(writer_ids)} writers")

    # Display writer information
    for writer_id in writer_ids:
        writer = wm.get_writer(writer_id)
        if writer:
            print(f"\n  Writer: {writer['name']}")
            print(f"    ID: {writer['writer_id']}")
            print(f"    Mentions: {writer['total_mentions']}")
            print(f"    Documents: {len(writer['writings'])}")

    return writer_ids


def test_byline_patterns():
    """Test various byline pattern formats"""
    print("\n" + "="*60)
    print("TEST 4: Byline Pattern Recognition")
    print("="*60)

    test_cases = [
        "By John Smith\nArticle content...",
        "Written by Jane Doe\nStory begins...",
        "STORY BY Michael Brown\nIn a surprising turn...",
        "Report by Sarah Williams and Tom Davis\nToday's events...",
        "بقلم محمد أحمد\nالتقرير يتحدث عن...",  # Arabic byline
    ]

    wm = WriterManager()

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Input: {text[:50]}...")
        writers = wm.extract_writer_names(text)
        print(f"  Found: {writers}")


def check_dependencies():
    """Check if all required dependencies are available"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)

    dependencies = {
        "PyMuPDF": False,
        "spaCy": False,
        "spaCy Model": False
    }

    # Check PyMuPDF
    try:
        import pymupdf
        dependencies["PyMuPDF"] = True
        print("✓ PyMuPDF is installed")
    except ImportError:
        print("✗ PyMuPDF is NOT installed")
        print("  Install with: pip install pymupdf")

    # Check spaCy
    try:
        import spacy
        dependencies["spaCy"] = True
        print("✓ spaCy is installed")

        # Check model
        try:
            nlp = spacy.load("en_core_web_sm")
            dependencies["spaCy Model"] = True
            print("✓ spaCy model 'en_core_web_sm' is available")
        except OSError:
            print("✗ spaCy model 'en_core_web_sm' is NOT available")
            print("  Download with: python -m spacy download en_core_web_sm")
    except ImportError:
        print("✗ spaCy is NOT installed")
        print("  Install with: pip install spacy")

    print("\n" + "-"*60)
    all_ready = all(dependencies.values())
    if all_ready:
        print("✓ All dependencies are ready!")
    else:
        print("⚠ Some dependencies are missing. Enhanced extraction may not work fully.")

    return all_ready


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Enhanced Writer/Journalist Extraction - Test Suite")
    print("="*60)

    # Check dependencies first
    deps_ok = check_dependencies()

    # Run basic tests
    test_basic_extraction()
    test_byline_patterns()
    test_process_document()

    # Test PDF extraction if path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        test_pdf_extraction(pdf_path)
    else:
        print("\n" + "="*60)
        print("PDF Test Skipped")
        print("="*60)
        print("To test PDF extraction, provide a PDF path:")
        print("  python test_enhanced_extraction.py /path/to/test.pdf")

    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
