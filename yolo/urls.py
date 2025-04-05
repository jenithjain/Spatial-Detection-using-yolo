from django.urls import path
from . import views

app_name = 'yolo'

urlpatterns = [
    path('', views.upload_page, name='upload'),
    path('process-image/', views.process_image, name='process_image'),
    path('compare/', views.compare_detections, name='compare'),
    path('compare/<str:session_id>/', views.compare_detections, name='compare_with_id'),
    path('api/comparison/', views.get_comparison_json, name='get_comparison'),
] 