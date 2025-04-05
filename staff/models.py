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
