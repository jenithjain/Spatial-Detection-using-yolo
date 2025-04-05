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
] 