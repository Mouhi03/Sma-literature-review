"""Dashboard URL configuration."""

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('corpus/', views.corpus, name='corpus'),
    path('corpus/<str:doc_id>/', views.article_detail, name='article_detail'),
    path('themes/', views.themes, name='themes'),
    path('revue/', views.revue, name='revue'),
    path('revue/export/latex/', views.export_latex, name='export_latex'),
    path('gaps/', views.gaps, name='gaps'),
    path('citations/', views.citations, name='citations'),
    path('pipeline/', views.pipeline, name='pipeline'),
    path('upload/', views.upload_pdfs, name='upload'),
    path('api/pipeline/run/', views.api_run_pipeline, name='api_run_pipeline'),
]
