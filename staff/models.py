from django.db import models
from django.contrib.auth.models import User
from manager.models import Manager

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
    check_in_time = models.DateTimeField(auto_now_add=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='active')
    notes = models.TextField(blank=True, null=True)
    yolo_session_id = models.CharField(max_length=100, blank=True, null=True)
    has_missing_items = models.BooleanField(default=False)
    missing_items_details = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Room {self.room_number} - {self.status} - {self.check_in_time.date()}"
    
    class Meta:
        ordering = ['-check_in_time']
        verbose_name_plural = "Room Activities"
