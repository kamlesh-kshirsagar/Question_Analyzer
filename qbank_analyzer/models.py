from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class QuestionBank(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(default=timezone.now)
    pdf_file = models.FileField(upload_to='question_banks/')
    
    def __str__(self):
        return self.title

class Question(models.Model):
    question_bank = models.ForeignKey(QuestionBank, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    topic = models.CharField(max_length=100)
    difficulty_level = models.CharField(max_length=20, choices=[
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard')
    ])
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.topic} - {self.question_text[:50]}..."

class QuestionAnalysis(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='analyses')
    frequency = models.FloatField(default=0.0)  # Percentage of occurrence
    trend = models.CharField(max_length=20, choices=[
        ('increasing', 'Increasing'),
        ('decreasing', 'Decreasing'),
        ('stable', 'Stable')
    ])
    suggested_answer = models.TextField(blank=True)
    confidence_score = models.FloatField(default=0.0)
    last_analyzed = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Analysis for {self.question.topic} - {self.frequency}%"

class TopicSearch(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    search_text = models.TextField()
    search_date = models.DateTimeField(default=timezone.now)
    results_count = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Search: {self.search_text[:50]}..."
