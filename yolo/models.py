from django.db import models
import json
from staff.models import StaffMember

# Create your models here.
class Detection(models.Model):
    DETECTION_TYPE_CHOICES = (
        ('checkin', 'Check-in Image'),
        ('checkout', 'Check-out Image'),
    )
    
    image = models.ImageField(upload_to='detections/')
    processed_image = models.ImageField(upload_to='processed_detections/', null=True, blank=True)
    detection_type = models.CharField(max_length=10, choices=DETECTION_TYPE_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)
    detection_data = models.TextField()  # JSON data of detected objects
    session_id = models.CharField(max_length=100)  # To group related checkin and checkout images
    staff_member = models.ForeignKey(StaffMember, on_delete=models.CASCADE, null=True, blank=True)
    room_number = models.CharField(max_length=20, blank=True, null=True)
    
    def set_detection_data(self, data):
        self.detection_data = json.dumps(data)
    
    def get_detection_data(self):
        return json.loads(self.detection_data)
    
    def __str__(self):
        return f"{self.detection_type} Detection - Room {self.room_number} - {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']
