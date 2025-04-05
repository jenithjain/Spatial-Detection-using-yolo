from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import ManagerRegistrationForm
from .models import Manager

# Create your views here.

def manager_register(request):
    if request.method == 'POST':
        form = ManagerRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful. Please login.')
            return redirect('login')
    else:
        form = ManagerRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def manager_dashboard(request):
    try:
        manager = request.user.manager
        staff_members = manager.staffmember_set.all()
        context = {
            'manager': manager,
            'staff_members': staff_members,
        }
        return render(request, 'admin/dashboard.html', context)
    except Manager.DoesNotExist:
        messages.error(request, 'Access denied. Manager profile not found.')
        return redirect('login')
