from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Manager(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    organization_name = models.CharField(max_length=100)
    contact_number = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.organization_name}"

class Room(models.Model):
    manager = models.ForeignKey(Manager, on_delete=models.CASCADE)
    room_number = models.CharField(max_length=10)
    room_type = models.CharField(max_length=50, default='Standard')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Room {self.room_number} - {self.room_type}"
    
    class Meta:
        unique_together = ['manager', 'room_number']

class RoomInventory(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE, related_name='inventory_items')
    item_name = models.CharField(max_length=100)
    quantity = models.PositiveIntegerField(default=1)
    description = models.TextField(blank=True, null=True)
    added_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.quantity} {self.item_name}(s) in Room {self.room.room_number}"
    
    class Meta:
        verbose_name_plural = "Room Inventories"
