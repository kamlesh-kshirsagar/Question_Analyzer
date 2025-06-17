#!/usr/bin/env python3
"""
Test script to verify Google Gemini API key is working
"""

import os
import django
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'question_analyzer.settings')
django.setup()

import google.generativeai as genai

def test_gemini_connection():
    """Test Google Gemini API connection"""
    try:
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test with a simple request
        response = model.generate_content("Hello! Can you respond with 'Gemini API is working' if you receive this message?")
        
        print("✅ Google Gemini API Test Successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Google Gemini API Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Google Gemini API connection...")
    test_gemini_connection() 