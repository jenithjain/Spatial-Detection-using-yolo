from django.db import models
from django.contrib.auth.models import User
from manager.models import Manager, Room
import json

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

class CheckoutAnalysis(models.Model):
    """Model to store detailed visual analysis results from room checkouts"""
    ANALYSIS_STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    room_activity = models.ForeignKey(RoomActivity, on_delete=models.CASCADE, related_name='checkout_analyses')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=15, choices=ANALYSIS_STATUS_CHOICES, default='pending')
    
    # Original images
    checkin_image = models.ImageField(upload_to='checkout_analysis/originals/checkin/', null=True, blank=True)
    checkout_image = models.ImageField(upload_to='checkout_analysis/originals/checkout/', null=True, blank=True)
    
    # Analysis method results
    yolo_result = models.TextField(blank=True, null=True, help_text="JSON data from YOLO object detection comparison")
    spatial_analysis_result = models.TextField(blank=True, null=True, help_text="JSON data from spatial analysis")
    feature_matching_result = models.TextField(blank=True, null=True, help_text="JSON data from feature matching")
    color_histogram_result = models.TextField(blank=True, null=True, help_text="JSON data from color histogram analysis")
    heatmap_result = models.TextField(blank=True, null=True, help_text="JSON data from heatmap analysis")
    damage_detection_result = models.TextField(blank=True, null=True, help_text="JSON data from damage detection analysis")
    
    # Processed/visualization images
    yolo_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/yolo/', null=True, blank=True)
    spatial_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/spatial/', null=True, blank=True)
    feature_matching_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/feature_matching/', null=True, blank=True)
    color_histogram_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/color_histogram/', null=True, blank=True)
    heatmap_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/heatmap/', null=True, blank=True)
    damage_visualization = models.ImageField(upload_to='checkout_analysis/visualizations/damage/', null=True, blank=True)
    
    # Summary
    summary_result = models.TextField(blank=True, null=True, help_text="JSON summary of all analysis methods")
    confidence_score = models.FloatField(default=0.0, help_text="Overall confidence score for the analysis")
    
    # Human-readable summaries
    simple_summary = models.TextField(blank=True, null=True, help_text="Simple human-readable summary of the analysis")
    recommendations = models.TextField(blank=True, null=True, help_text="Recommendations based on the analysis")
    
    def __str__(self):
        return f"Checkout Analysis for Room {self.room_activity.room_number} - {self.created_at.date()}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Checkout Analyses"
    
    def get_yolo_result(self):
        """Get YOLO result as Python object"""
        if not self.yolo_result:
            return {}
        try:
            return json.loads(self.yolo_result)
        except Exception as e:
            print(f"Error parsing YOLO result: {e}")
            return {'error': 'Could not parse YOLO result'}
    
    def get_spatial_analysis_result(self):
        """Get spatial analysis result as Python object"""
        if not self.spatial_analysis_result:
            return {}
        try:
            return json.loads(self.spatial_analysis_result)
        except Exception as e:
            print(f"Error parsing spatial analysis result: {e}")
            return {'error': 'Could not parse spatial analysis result'}
    
    def get_feature_matching_result(self):
        """Get feature matching result as Python object"""
        if not self.feature_matching_result:
            return {}
        try:
            return json.loads(self.feature_matching_result)
        except Exception as e:
            print(f"Error parsing feature matching result: {e}")
            return {'error': 'Could not parse feature matching result'}
    
    def get_color_histogram_result(self):
        """Get color histogram result as Python object"""
        if not self.color_histogram_result:
            return {}
        try:
            return json.loads(self.color_histogram_result)
        except Exception as e:
            print(f"Error parsing color histogram result: {e}")
            return {'error': 'Could not parse color histogram result'}
            
    def get_heatmap_result(self):
        """Get heatmap result as Python object"""
        if not self.heatmap_result:
            return {}
        try:
            return json.loads(self.heatmap_result)
        except Exception as e:
            print(f"Error parsing heatmap result: {e}")
            return {'error': 'Could not parse heatmap result'}
            
    def get_damage_detection_result(self):
        """Get damage detection result as Python object"""
        if not self.damage_detection_result:
            return {}
        try:
            return json.loads(self.damage_detection_result)
        except Exception as e:
            print(f"Error parsing damage detection result: {e}")
            return {'error': 'Could not parse damage detection result'}
    
    def get_summary_result(self):
        """Get summary result as Python object"""
        if not self.summary_result:
            return {}
        try:
            return json.loads(self.summary_result)
        except Exception as e:
            print(f"Error parsing summary result: {e}")
            return {'error': 'Could not parse summary result'}
