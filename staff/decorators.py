from django.shortcuts import redirect
from django.contrib import messages
from functools import wraps

def staff_required(view_func):
    """
    Decorator for views that checks if the user is a staff member.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Import here to avoid circular import issues
        from .models import StaffMember
        
        try:
            staff_member = StaffMember.objects.get(user=request.user)
            return view_func(request, *args, **kwargs)
        except StaffMember.DoesNotExist:
            messages.error(request, 'Access denied. Staff profile not found.')
            return redirect('staff:login')
    return wrapper 