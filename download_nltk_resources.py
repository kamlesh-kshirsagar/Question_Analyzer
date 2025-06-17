#!/usr/bin/env python3
"""
Script to download all required NLTK resources for the qbank project.
Run this script once to ensure all NLTK resources are available.
"""

import nltk
import sys

def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = [
        'punkt',
        'stopwords',
        'punkt_tab'
    ]
    
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=False)
            print(f"✓ {resource} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {resource}: {e}")
            return False
    
    print("All NLTK resources downloaded successfully!")
    return True

if __name__ == "__main__":
    success = download_nltk_resources()
    sys.exit(0 if success else 1) 