from django.urls import path
from . import views

app_name = 'qbank_analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_question_bank, name='upload'),
    path('analyze/<int:bank_id>/', views.analyze_questions, name='analyze'),
    path('search/', views.search_topics, name='search'),
] 