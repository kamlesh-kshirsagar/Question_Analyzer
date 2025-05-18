from django import forms
from .models import QuestionBank, TopicSearch

class QuestionBankUploadForm(forms.ModelForm):
    class Meta:
        model = QuestionBank
        fields = ['title', 'description', 'pdf_file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
        }

class TopicSearchForm(forms.ModelForm):
    class Meta:
        model = TopicSearch
        fields = ['search_text']
        widgets = {
            'search_text': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter topic or keywords to search...'
            })
        } 