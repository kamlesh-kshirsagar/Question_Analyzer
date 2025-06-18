from django import forms

class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(label="Upload PDF", required=True)
    user_input = forms.CharField(label="Enter topic/question", max_length=200, required=True) 