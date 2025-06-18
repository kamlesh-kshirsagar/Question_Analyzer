from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('generate-answer/', views.generate_answer, name='generate_answer'),
    path('find-similar-questions/', views.find_similar_questions, name='find_similar_questions'),
] 