from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'staff'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(
        template_name='staff/registration/login.html',
        redirect_authenticated_user=True
    ), name='login'),
    path('register/', views.staff_register, name='register'),
    path('dashboard/', views.staff_dashboard, name='dashboard'),
    
    # Room check-in/check-out URLs
    path('room/check-in/', views.room_check_in, name='room_check_in'),
    path('room/check-in/upload/<int:activity_id>/', views.upload_checkin_image, name='upload_checkin_image'),
    path('room/check-out/<int:activity_id>/', views.room_check_out, name='room_check_out'),
    path('room/check-out/upload/<int:activity_id>/', views.upload_checkout_image, name='upload_checkout_image'),
    path('room/comparison/<int:activity_id>/', views.view_comparison, name='view_comparison'),
    path('room/mark-as-ok/<int:activity_id>/', views.mark_item_as_ok, name='mark_item_as_ok'),
    path('room/detailed-analysis/<int:activity_id>/', views.detailed_checkout_analysis, name='detailed_checkout_analysis'),
    path('room/analyze-checkout/<int:activity_id>/', views.analyze_checkout_images, name='analyze_checkout_images'),
    path('api/analyze-checkout/', views.analyze_checkout, name='api_analyze_checkout'),
] 