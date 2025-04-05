from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import ManagerRegistrationForm
from .models import Manager, Room, RoomInventory

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
        # Get all rooms for the current manager with their inventory
        rooms = Room.objects.filter(manager=manager)
        rooms_with_inventory = []
        
        # For each room, get its inventory items
        for room in rooms:
            inventory_items = room.inventory_items.all()
            rooms_with_inventory.append({
                'room': room,
                'inventory': inventory_items
            })
            
        context = {
            'manager': manager,
            'staff_members': staff_members,
            'rooms_with_inventory': rooms_with_inventory,
        }
        return render(request, 'admin/dashboard.html', context)
    except Manager.DoesNotExist:
        messages.error(request, 'Access denied. Manager profile not found.')
        return redirect('login')
