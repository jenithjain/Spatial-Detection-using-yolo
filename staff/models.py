from django.db import models
from django.contrib.auth.models import User
from manager.models import Manager, Room

# Create your models here.

class StaffMember(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    manager = models.ForeignKey(Manager, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, default='')
    role = models.CharField(max_length=50, default='Staff')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} - {self.role}"

class RoomActivity(models.Model):
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('completed', 'Completed'),
    )
    
    staff_member = models.ForeignKey(StaffMember, on_delete=models.CASCADE)
    room_number = models.CharField(max_length=20)
    room = models.ForeignKey(Room, on_delete=models.SET_NULL, null=True, blank=True)
    check_in_time = models.DateTimeField(auto_now_add=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='active')
    notes = models.TextField(blank=True, null=True)
    yolo_session_id = models.CharField(max_length=100, blank=True, null=True)
    has_missing_items = models.BooleanField(default=False)
    missing_items_details = models.TextField(blank=True, null=True)
    added_items = models.TextField(blank=True, null=True)
    shifted_items = models.TextField(blank=True, null=True)
    approved_items = models.TextField(blank=True, null=True, help_text="JSON list of items that have been manually marked as OK")
    
    def __str__(self):
        return f"Room {self.room_number} - {self.status} - {self.check_in_time.date()}"
    
    class Meta:
        ordering = ['-check_in_time']
        verbose_name_plural = "Room Activities"

    def get_missing_items(self):
        """Safely get missing items as a dictionary for template use"""
        if not self.missing_items_details:
            return {}
        
        try:
            import json
            missing_items = json.loads(self.missing_items_details)
            
            # If it's already a dict, just return it
            if isinstance(missing_items, dict):
                return missing_items
                
            # If it's a list, try to convert to dict
            if isinstance(missing_items, list):
                result = {}
                for item in missing_items:
                    if isinstance(item, dict) and len(item) == 1:
                        key, value = next(iter(item.items()))
                        result[key] = value
                    else:
                        # Handle simple strings or other formats
                        result[str(item)] = 1
                return result
            
            # If it's a string or other type, return a dict with that as a key
            return {str(missing_items): 1}
            
        except Exception as e:
            print(f"Error parsing missing items: {e}")
            return {'error': 'Could not parse missing items'}
