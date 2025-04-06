from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'manager'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(
        template_name='manager/registration/login.html',
        redirect_authenticated_user=True,
        next_page='manager:dashboard'
    ), name='login'),
    path('register/', views.manager_register, name='register'),
    path('dashboard/', views.manager_dashboard, name='dashboard'),
    path('inventory/', views.manager_inventory, name='inventory'),
    path('analysis/', views.manager_analysis, name='analysis'),
] 