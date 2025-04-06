from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count, Q, Avg
from .forms import ManagerRegistrationForm
from .models import Manager, Room, RoomInventory
from staff.models import RoomActivity, CheckoutAnalysis

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
        rooms_with_data = []
        
        # For each room, get its inventory items and analysis data
        for room in rooms:
            # Get inventory items
            inventory_items = room.inventory_items.all()
            
            # Get room activities for this room
            room_activities = RoomActivity.objects.filter(room=room)
            
            # Count check-ins and check-outs
            total_checkins = room_activities.count()
            total_checkouts = room_activities.filter(status='completed').count()
            
            # Get missing items statistics
            missing_items_count = room_activities.filter(has_missing_items=True).count()
            missing_items_percentage = (missing_items_count / total_checkouts * 100) if total_checkouts > 0 else 0
            
            # Get detailed analysis data
            analyses = CheckoutAnalysis.objects.filter(room_activity__room=room)
            
            # Calculate average confidence score
            avg_confidence = analyses.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
            
            # Get latest analysis
            latest_analysis = analyses.order_by('-created_at').first()
            
            # Get common issues
            common_issues = []
            damage_types = []
            
            for analysis in analyses:
                if analysis.status == 'completed' and analysis.summary_result:
                    try:
                        import json
                        summary = json.loads(analysis.summary_result)
                        
                        # Extract missing items from YOLO results
                        if 'yolo' in summary and 'items' in summary['yolo'] and 'missing' in summary['yolo']['items']:
                            common_issues.extend(summary['yolo']['items']['missing'])
                            
                        # Extract damage types
                        if 'damage' in summary and 'damage_types' in summary['damage']:
                            damage_types.extend(summary['damage']['damage_types'])
                    except:
                        pass
            
            # Count frequencies of issues and damages
            from collections import Counter
            common_issues_count = Counter(common_issues)
            damage_types_count = Counter(damage_types)
            
            # Get top 3 most common issues and damages
            top_issues = common_issues_count.most_common(3)
            top_damages = damage_types_count.most_common(3)
            
            rooms_with_data.append({
                'room': room,
                'inventory': inventory_items,
                'total_checkins': total_checkins,
                'total_checkouts': total_checkouts,
                'missing_items_count': missing_items_count,
                'missing_items_percentage': missing_items_percentage,
                'avg_confidence_score': avg_confidence,
                'latest_analysis': latest_analysis,
                'top_issues': top_issues,
                'top_damages': top_damages,
                'activities': room_activities.order_by('-check_in_time')[:5]  # Get 5 most recent activities
            })
            
        # Get aggregated statistics
        total_rooms = rooms.count()
        total_activities = RoomActivity.objects.filter(room__manager=manager).count()
        total_missing_items = RoomActivity.objects.filter(
            room__manager=manager, 
            has_missing_items=True
        ).count()
        
        context = {
            'manager': manager,
            'staff_members': staff_members,
            'rooms_with_data': rooms_with_data,
            'total_rooms': total_rooms,
            'total_activities': total_activities,
            'total_missing_items': total_missing_items,
        }
        return render(request, 'admin/dashboard.html', context)
    except Manager.DoesNotExist:
        messages.error(request, 'Access denied. Manager profile not found.')
        return redirect('login')

@login_required
def manager_inventory(request):
    try:
        manager = request.user.manager
        # Get all rooms for the current manager with their inventory
        rooms = Room.objects.filter(manager=manager)
        rooms_with_data = []
        
        # For each room, get its inventory items and analysis data
        for room in rooms:
            # Get inventory items
            inventory_items = room.inventory_items.all()
            
            # Get room activities for this room
            room_activities = RoomActivity.objects.filter(room=room)
            
            # Count check-ins and check-outs
            total_checkins = room_activities.count()
            total_checkouts = room_activities.filter(status='completed').count()
            
            # Get missing items statistics
            missing_items_count = room_activities.filter(has_missing_items=True).count()
            missing_items_percentage = (missing_items_count / total_checkouts * 100) if total_checkouts > 0 else 0
            
            # Get common issues for status indicators
            common_issues = []
            
            for activity in room_activities:
                analysis = activity.checkout_analyses.first()
                if analysis and analysis.status == 'completed' and analysis.summary_result:
                    try:
                        import json
                        summary = json.loads(analysis.summary_result)
                        
                        # Extract missing items from YOLO results
                        if 'yolo' in summary and 'items' in summary['yolo'] and 'missing' in summary['yolo']['items']:
                            common_issues.extend(summary['yolo']['items']['missing'])
                    except:
                        pass
            
            # Count frequencies of issues
            from collections import Counter
            common_issues_count = Counter(common_issues)
            
            # Get top 3 most common issues
            top_issues = common_issues_count.most_common(3)
            
            rooms_with_data.append({
                'room': room,
                'inventory': inventory_items,
                'missing_items_count': missing_items_count,
                'top_issues': top_issues
            })
        
        context = {
            'manager': manager,
            'rooms_with_data': rooms_with_data,
            'page_title': 'Room Inventory Management',
            'standalone_page': True
        }
        return render(request, 'admin/inventory_page.html', context)
    except Manager.DoesNotExist:
        messages.error(request, 'Access denied. Manager profile not found.')
        return redirect('login')

@login_required
def manager_analysis(request):
    try:
        manager = request.user.manager
        # Get all rooms for the current manager with their inventory
        rooms = Room.objects.filter(manager=manager)
        rooms_with_data = []
        
        # For each room, get its inventory items and analysis data
        for room in rooms:
            # Get inventory items
            inventory_items = room.inventory_items.all()
            
            # Get room activities for this room
            room_activities = RoomActivity.objects.filter(room=room)
            
            # Count check-ins and check-outs
            total_checkins = room_activities.count()
            total_checkouts = room_activities.filter(status='completed').count()
            
            # Get missing items statistics
            missing_items_count = room_activities.filter(has_missing_items=True).count()
            missing_items_percentage = (missing_items_count / total_checkouts * 100) if total_checkouts > 0 else 0
            
            # Get detailed analysis data
            analyses = CheckoutAnalysis.objects.filter(room_activity__room=room)
            
            # Calculate average confidence score
            avg_confidence = analyses.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
            
            # Get latest analysis
            latest_analysis = analyses.order_by('-created_at').first()
            
            # Get common issues
            common_issues = []
            damage_types = []
            
            for analysis in analyses:
                if analysis.status == 'completed' and analysis.summary_result:
                    try:
                        import json
                        summary = json.loads(analysis.summary_result)
                        
                        # Extract missing items from YOLO results
                        if 'yolo' in summary and 'items' in summary['yolo'] and 'missing' in summary['yolo']['items']:
                            common_issues.extend(summary['yolo']['items']['missing'])
                            
                        # Extract damage types
                        if 'damage' in summary and 'damage_types' in summary['damage']:
                            damage_types.extend(summary['damage']['damage_types'])
                    except:
                        pass
            
            # Count frequencies of issues and damages
            from collections import Counter
            common_issues_count = Counter(common_issues)
            damage_types_count = Counter(damage_types)
            
            # Get top 3 most common issues and damages
            top_issues = common_issues_count.most_common(3)
            top_damages = damage_types_count.most_common(3)
            
            rooms_with_data.append({
                'room': room,
                'inventory': inventory_items,
                'total_checkins': total_checkins,
                'total_checkouts': total_checkouts,
                'missing_items_count': missing_items_count,
                'missing_items_percentage': missing_items_percentage,
                'avg_confidence_score': avg_confidence,
                'latest_analysis': latest_analysis,
                'top_issues': top_issues,
                'top_damages': top_damages,
                'activities': room_activities.order_by('-check_in_time')[:5]  # Get 5 most recent activities
            })
            
        # Get aggregated statistics
        total_rooms = rooms.count()
        total_activities = RoomActivity.objects.filter(room__manager=manager).count()
        total_missing_items = RoomActivity.objects.filter(
            room__manager=manager, 
            has_missing_items=True
        ).count()
        
        context = {
            'manager': manager,
            'rooms_with_data': rooms_with_data,
            'total_rooms': total_rooms,
            'total_activities': total_activities,
            'total_missing_items': total_missing_items,
            'page_title': 'Room Analysis Dashboard',
            'standalone_page': True
        }
        return render(request, 'admin/analysis_page.html', context)
    except Manager.DoesNotExist:
        messages.error(request, 'Access denied. Manager profile not found.')
        return redirect('login')
