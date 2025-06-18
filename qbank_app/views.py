from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from .forms import PDFUploadForm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import PyPDF2
import nltk
import random
import os
import google.generativeai as genai

# Download required NLTK resources with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")

def ensure_nltk_resources():
    """Ensure all required NLTK resources are available"""
    try:
        # Check if punkt is available
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        # Check if stopwords is available
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        # Check if punkt_tab is available
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

# Ensure resources are available
ensure_nltk_resources()

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Ensure NLTK resources are available before processing
    ensure_nltk_resources()
    
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    try:
        words = word_tokenize(text)
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)
    except LookupError as e:
        print(f"NLTK resource error: {e}")
        # Fallback to simple word splitting if NLTK fails
        words = text.split()
        return " ".join(words)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text_with_pages = []
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        if text:
            text_with_pages.append((text.strip(), page_num))
    return text_with_pages

def find_similar_sentences(input_text, text_with_pages):
    all_sentences = []
    for text, page_num in text_with_pages:
        try:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                all_sentences.append((sentence, page_num))
        except LookupError as e:
            print(f"NLTK resource error in sentence tokenization: {e}")
            # Fallback to simple sentence splitting
            sentences = text.split('.')
            for sentence in sentences:
                if sentence.strip():
                    all_sentences.append((sentence.strip(), page_num))
    
    processed_sentences = [preprocess(sentence) for sentence, _ in all_sentences]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_sentences + [input_text])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # Filter matches to only include sentences with more than 30% similarity
    matches = [(all_sentences[i][0], all_sentences[i][1], similarity_scores[i] * 100) 
               for i in range(len(all_sentences)) 
               if similarity_scores[i] * 100 > 30]  # Only include matches > 30%
    
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches

def generate_ai_answer(question, context_sentences, pdf_text):
    """
    Generate an AI-powered answer using Google Gemini based on the question and context from the PDF
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare context from PDF sentences
        context_text = ""
        if context_sentences:
            context_text = "Context from the document:\n" + "\n".join([sentence for sentence, _, _ in context_sentences[:5]])
        
        # Create the prompt for Gemini
        prompt = f"""
You are an expert tutor and educator. Please provide a comprehensive and accurate answer to the following question in short.

{context_text}

Question: {question}

Please provide a detailed, well-structured answer that:
1. Directly addresses the question
2. Uses information from the provided context when relevant
3. Is educational and informative
4. Is written in a clear, professional tone
5. Includes relevant examples or explanations when helpful

Answer:"""

        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        # Extract the generated answer
        ai_answer = response.text.strip()
        
        # Calculate confidence based on response quality
        confidence = min(95, 70 + len(ai_answer.split()) // 2)  # Higher confidence for longer, more detailed answers
        
        return {
            'answer': ai_answer,
            'confidence': confidence,
            'context_used': len(context_sentences),
            'question_type': 'ai_generated',
            'model': 'gemini-1.5-flash'
        }
        
    except Exception as e:
        # Check for specific error types
        error_message = str(e)
        if "quota" in error_message.lower() or "429" in error_message:
            fallback_answer = "I apologize, but the AI service is currently experiencing high usage or quota limits. Please try again later or contact support if this persists. You may need to add credits to your Gemini account."
        elif "api_key" in error_message.lower():
            fallback_answer = "I apologize, but there's an issue with the AI service configuration. Please contact support."
        else:
            fallback_answer = f"I apologize, but I'm unable to generate an AI answer at the moment due to a technical issue: {str(e)}. Please try again later or contact support if the problem persists."
        
        return {
            'answer': fallback_answer,
            'confidence': 30,
            'context_used': 0,
            'question_type': 'fallback',
            'model': 'fallback'
        }

def generate_fallback_answer(question, context_sentences):
    """
    Generate a fallback answer when OpenAI API is not available
    """
    question_lower = question.lower()
    
    # Simple answer templates based on question type
    if any(word in question_lower for word in ['what', 'define', 'explain']):
        if context_sentences:
            context = context_sentences[0][0] if context_sentences else ""
            return f"Based on the provided context, this appears to be related to: {context[:100]}... For a more detailed answer, please try again later when the AI service is available."
        else:
            return "This appears to be a definition or explanation question. For a comprehensive answer, please try again later when the AI service is available."
    
    elif any(word in question_lower for word in ['how', 'process', 'method']):
        return "This appears to be a process or method question. The answer would typically involve step-by-step instructions or procedures. For a detailed explanation, please try again later when the AI service is available."
    
    elif any(word in question_lower for word in ['why', 'reason', 'cause']):
        return "This appears to be asking about reasons or causes. The answer would typically involve explaining the underlying factors or motivations. For a comprehensive analysis, please try again later when the AI service is available."
    
    elif any(word in question_lower for word in ['when', 'time', 'date']):
        return "This appears to be asking about timing or dates. For specific temporal information, please try again later when the AI service is available."
    
    else:
        return "This is an interesting question that would benefit from a detailed AI-generated response. Please try again later when the AI service is available for a comprehensive answer."

def user_login(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, 'qbank_app/login.html')

def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
def home(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf_file']
            user_input = form.cleaned_data['user_input']
            processed_input = preprocess(user_input)
            text_with_pages = extract_text_from_pdf(pdf_file)
            matches = find_similar_sentences(processed_input, text_with_pages)
            
            # Extract all text from PDF for question extraction
            all_pdf_text = ' '.join([text for text, _ in text_with_pages])
            extracted_questions = extract_questions_using_gemini(all_pdf_text)
            
            # Find similar questions using Gemini
            similar_questions = find_similar_questions_using_gemini(user_input, extracted_questions)
            
            # Store PDF context in session for AI answer generation
            pdf_context = '|'.join([sentence for sentence, _, _ in matches[:10]])
            request.session['pdf_context'] = pdf_context
            request.session['user_question'] = user_input
            request.session['extracted_questions'] = extracted_questions
            request.session['similar_questions'] = similar_questions
            
            return render(request, 'qbank_app/results.html', {
                'matches': matches,
                'user_input': user_input,
                'pdf_context': pdf_context,
                'extracted_questions': extracted_questions,
                'similar_questions': similar_questions
            })
    else:
        form = PDFUploadForm()
    return render(request, 'qbank_app/home.html', {'form': form})

@login_required
def generate_answer(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')
        
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        try:
            # Get PDF context from session
            pdf_context = request.session.get('pdf_context', '')
            context_sentences = []
            
            if pdf_context:
                # Parse context data from session
                sentences = pdf_context.split('|')
                for i, sentence in enumerate(sentences[:5]):
                    if sentence.strip():
                        context_sentences.append((sentence.strip(), 1, 80 - i * 10))
            
            # Generate AI answer
            ai_response = generate_ai_answer(question, context_sentences, "")
            
            return JsonResponse({
                'success': True,
                'answer': ai_response['answer'],
                'confidence': ai_response['confidence'],
                'context_used': ai_response['context_used'],
                'question_type': ai_response['question_type'],
                'model': ai_response['model']
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Error generating answer: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def extract_questions_from_text(text):
    """
    Extract questions from text using NLTK and regex patterns
    """
    questions = []
    
    # Split text into sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback to simple sentence splitting
        sentences = text.split('.')
    
    # Question patterns
    question_patterns = [
        r'\b(what|how|why|when|where|who|which|whom|whose)\b.*\?',
        r'.*\?$',  # Any sentence ending with ?
        r'\b(explain|describe|define|compare|analyze|discuss|evaluate|summarize)\b.*',
        r'\b(can you|could you|would you|please)\b.*',
        r'\b(tell me about|what is|what are|how does|why does)\b.*'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter out very short sentences
            for pattern in question_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    questions.append(sentence)
                    break
    
    return questions[:10]  # Return top 10 questions

def extract_questions_using_gemini(text):
    """
    Extract questions from text using Gemini AI for better accuracy
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for question extraction
        prompt = f"""
You are an expert at extracting questions from educational text. Please analyze the following text and extract all potential questions that could be asked about the content.

Text to analyze:
{text[:4000]}  # Limit text length to avoid token limits

Please extract questions that:
1. Are directly related to the content
2. Cover different aspects (definitions, processes, comparisons, etc.)
3. Are clear and well-formed
4. Could be used for educational purposes

Return only the questions, one per line, without numbering or additional text.
If no questions can be extracted, return "No questions found."

Questions:"""

        response = model.generate_content(prompt)
        questions_text = response.text.strip()
        
        if "No questions found" in questions_text:
            return []
        
        # Split into individual questions and clean them
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('Questions:'):
                # Remove numbering if present
                line = re.sub(r'^\d+\.\s*', '', line)
                questions.append(line)
        
        return questions[:15]  # Return top 15 questions
        
    except Exception as e:
        print(f"Error extracting questions with Gemini: {e}")
        # Fallback to regex-based extraction
        return extract_questions_from_text(text)

def find_similar_questions_using_gemini(user_question, extracted_questions):
    """
    Find similar questions using Gemini AI for semantic similarity
    """
    try:
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for similarity analysis
        prompt = f"""
You are an expert at analyzing question similarity. Please analyze the similarity between the user's question and a list of extracted questions.

User Question: "{user_question}"

Extracted Questions:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(extracted_questions)])}

For each extracted question, provide:
1. A similarity score from 0-100 (where 100 is identical)
2. A brief explanation of why it's similar or different

Format your response as:
Question 1: [score]% - [explanation]
Question 2: [score]% - [explanation]
...

Focus on semantic similarity, not just keyword matching. Consider:
- Topic relevance
- Question type (what, how, why, etc.)
- Subject matter alignment
- Conceptual similarity"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the response to extract scores
        similar_questions = []
        lines = response_text.split('\n')
        
        for line in lines:
            if 'Question' in line and '%' in line:
                try:
                    # Extract question number and score
                    parts = line.split(':')
                    if len(parts) >= 2:
                        question_part = parts[0].strip()
                        score_part = parts[1].strip()
                        
                        # Extract question number
                        question_num_match = re.search(r'Question (\d+)', question_part)
                        if question_num_match:
                            question_num = int(question_num_match.group(1)) - 1
                            
                            # Extract score
                            score_match = re.search(r'(\d+)%', score_part)
                            if score_match and question_num < len(extracted_questions):
                                score = int(score_match.group(1))
                                if score > 20:  # Only include questions with >20% similarity
                                    similar_questions.append({
                                        'question': extracted_questions[question_num],
                                        'similarity_score': score,
                                        'explanation': score_part.split('-', 1)[1].strip() if '-' in score_part else ''
                                    })
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")
                    continue
        
        # Sort by similarity score (highest first)
        similar_questions.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_questions[:10]  # Return top 10 similar questions
        
    except Exception as e:
        print(f"Error finding similar questions with Gemini: {e}")
        # Fallback to simple keyword matching
        return find_similar_questions_fallback(user_question, extracted_questions)

def find_similar_questions_fallback(user_question, extracted_questions):
    """
    Fallback method for finding similar questions using keyword matching
    """
    user_words = set(preprocess(user_question).split())
    similar_questions = []
    
    for question in extracted_questions:
        question_words = set(preprocess(question).split())
        
        # Calculate Jaccard similarity
        intersection = len(user_words.intersection(question_words))
        union = len(user_words.union(question_words))
        
        if union > 0:
            similarity = (intersection / union) * 100
            if similarity > 10:  # Only include questions with >10% similarity
                similar_questions.append({
                    'question': question,
                    'similarity_score': round(similarity, 1),
                    'explanation': f"Keyword overlap: {intersection} common words"
                })
    
    similar_questions.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar_questions[:10]

@login_required
def find_similar_questions(request):
    """
    AJAX endpoint to find similar questions from the uploaded PDF
    """
    if request.method == 'POST':
        user_question = request.POST.get('question', '')
        
        if not user_question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        try:
            # Get extracted questions from session
            extracted_questions = request.session.get('extracted_questions', [])
            
            if not extracted_questions:
                return JsonResponse({'error': 'No questions found in the uploaded PDF'}, status=404)
            
            # Find similar questions using Gemini
            similar_questions = find_similar_questions_using_gemini(user_question, extracted_questions)
            
            return JsonResponse({
                'success': True,
                'similar_questions': similar_questions,
                'total_found': len(similar_questions),
                'user_question': user_question
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Error finding similar questions: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
