from django.urls import path
from . import views

app_name = 'yolo'

urlpatterns = [
    path('', views.upload_page, name='upload'),
    path('process-image/', views.process_image, name='process_image'),
] 