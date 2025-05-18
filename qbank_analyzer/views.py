from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count
from .models import QuestionBank, Question, QuestionAnalysis, TopicSearch
from .forms import QuestionBankUploadForm, TopicSearchForm
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from transformers import pipeline
import torch

# Initialize the AI model for answer suggestions
try:
    answer_generator = pipeline("text2text-generation", model="t5-base")
except:
    answer_generator = None

@login_required
def home(request):
    question_banks = QuestionBank.objects.filter(uploaded_by=request.user).order_by('-upload_date')
    search_form = TopicSearchForm()
    return render(request, 'qbank_analyzer/home.html', {
        'question_banks': question_banks,
        'search_form': search_form
    })

@login_required
def upload_question_bank(request):
    if request.method == 'POST':
        form = QuestionBankUploadForm(request.POST, request.FILES)
        if form.is_valid():
            question_bank = form.save(commit=False)
            question_bank.uploaded_by = request.user
            question_bank.save()
            
            # Process the PDF file
            try:
                process_pdf(question_bank)
                messages.success(request, 'Question bank uploaded and processed successfully!')
            except Exception as e:
                messages.error(request, f'Error processing PDF: {str(e)}')
            
            return redirect('home')
    else:
        form = QuestionBankUploadForm()
    
    return render(request, 'qbank_analyzer/upload.html', {'form': form})

def process_pdf(question_bank):
    pdf_file = question_bank.pdf_file
    reader = PdfReader(pdf_file)
    text = ""
    
    # Extract text from PDF
    for page in reader.pages:
        text += page.extract_text()
    
    # Basic question extraction (you might want to improve this based on your PDF format)
    questions = text.split('\n\n')
    
    for q_text in questions:
        if len(q_text.strip()) > 20:  # Basic filter for actual questions
            question = Question.objects.create(
                question_bank=question_bank,
                question_text=q_text.strip(),
                topic='General',  # You might want to implement topic detection
                difficulty_level='medium'
            )
            
            # Create initial analysis
            QuestionAnalysis.objects.create(
                question=question,
                frequency=0.0,
                trend='stable'
            )

@login_required
def analyze_questions(request, bank_id):
    question_bank = QuestionBank.objects.get(id=bank_id, uploaded_by=request.user)
    questions = Question.objects.filter(question_bank=question_bank)
    
    # Calculate frequencies and trends
    total_questions = questions.count()
    for question in questions:
        analysis = question.analyses.first()
        if analysis:
            # Calculate frequency (this is a simple example)
            analysis.frequency = (1 / total_questions) * 100
            
            # Generate suggested answer if AI model is available
            if answer_generator:
                try:
                    suggested = answer_generator(
                        f"Generate an answer for this question: {question.question_text}",
                        max_length=200,
                        num_return_sequences=1
                    )
                    analysis.suggested_answer = suggested[0]['generated_text']
                    analysis.confidence_score = 0.8  # This should be calculated based on model confidence
                except:
                    pass
            
            analysis.save()
    
    return render(request, 'qbank_analyzer/analysis.html', {
        'question_bank': question_bank,
        'questions': questions
    })

@login_required
def search_topics(request):
    if request.method == 'POST':
        form = TopicSearchForm(request.POST)
        if form.is_valid():
            search = form.save(commit=False)
            search.user = request.user
            
            # Perform the search
            questions = Question.objects.filter(
                question_text__icontains=search.search_text
            ) | Question.objects.filter(
                topic__icontains=search.search_text
            )
            
            search.results_count = questions.count()
            search.save()
            
            return render(request, 'qbank_analyzer/search_results.html', {
                'questions': questions,
                'search_text': search.search_text
            })
    return redirect('home')
