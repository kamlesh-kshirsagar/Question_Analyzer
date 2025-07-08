# QBank - Question Bank Analyzer

A Django web application that analyzes PDF documents and finds similar sentences based on user input using NLTK and Google's Gemini AI for intelligent answer generation.
## Features

- PDF text extraction and analysis
- Natural Language Processing with NLTK
- Similarity matching using TF-IDF and cosine similarity
- **AI-powered answer generation using Google Gemini**
- Question extraction from PDF content
- User authentication system
- Web-based interface

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment (if using one)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install Python packages
pip install -r requirements.txt
```

### 2. Download NLTK Resources

```bash
# Download required NLTK data
python download_nltk_resources.py
```

### 3. Configure Google Gemini API

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Update the `GEMINI_API_KEY` in `question_analyzer/settings.py`:

```python
GEMINI_API_KEY = 'your-actual-gemini-api-key-here'
```

### 4. Run Database Migrations

```bash
python manage.py migrate
```

### 5. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 6. Run the Development Server

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## Usage

1. Navigate to the home page
2. Upload a PDF document
3. Enter your search query
4. View extracted questions from the PDF
5. Click "AI Answer" buttons to generate intelligent answers using Gemini AI
6. Type custom questions and get AI-generated responses

## Technical Details

- **Framework**: Django 5.2+
- **NLP Library**: NLTK 3.8+
- **ML Library**: scikit-learn
- **PDF Processing**: PyPDF2
- **AI Service**: Google Gemini 1.5 Flash
- **Database**: SQLite (default)

## Troubleshooting

### Gemini API Issues

If you encounter Gemini API errors:

1. Verify your API key is correct in `question_analyzer/settings.py`
2. Check your Gemini API quota and billing status
3. Ensure you have enabled the Gemini API in your Google Cloud Console
4. Run the test script: `python test_openai.py` (renamed but still works)

### Common Issues

- **API Key Errors**: Check your Gemini API key configuration
- **Quota Exceeded**: Add credits to your Google AI Studio account
- **PDF upload errors**: Ensure the PDF file is not corrupted and is readable
- **Authentication issues**: Create a superuser account or check login credentials 
