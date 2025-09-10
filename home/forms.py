# forms.py
from django import forms
from .models import RecommendDocument,ListLesson

class DocumentUploadForm(forms.ModelForm):
    class Meta:
        model = RecommendDocument
        fields = ['title', 'author', 'source', 'url', 'format']

class VideoForm(forms.ModelForm):
    class Meta:
        model = ListLesson
        fields = ['lesson_name', 'description', 'video_file']

