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

class ModelValidation(models.Model):
    """Model to store model validation results for showcase purposes"""
    
    VALIDATION_TYPE_CHOICES = (
        ('object_detection', 'Object Detection'),
        ('gemini_comparison', 'Room Comparison'),
        ('damage_detection', 'Damage Detection'),
    )
    
    staff_member = models.ForeignKey(StaffMember, on_delete=models.CASCADE, null=True, blank=True)
    validation_type = models.CharField(max_length=30, choices=VALIDATION_TYPE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=100, blank=True, null=True, help_text="Optional name for this validation")
    description = models.TextField(blank=True, null=True)
    
    # Store the uploaded and processed images
    checkin_image = models.ImageField(upload_to='model_validation/checkin/', null=True, blank=True)
    checkout_image = models.ImageField(upload_to='model_validation/checkout/', null=True, blank=True)
    checkin_annotated = models.ImageField(upload_to='model_validation/checkin_annotated/', null=True, blank=True) 
    checkout_annotated = models.ImageField(upload_to='model_validation/checkout_annotated/', null=True, blank=True)
    
    # Analysis results
    checkin_objects = models.TextField(blank=True, null=True, help_text="JSON data of detected objects in checkin image")
    checkout_objects = models.TextField(blank=True, null=True, help_text="JSON data of detected objects in checkout image")
    missing_items = models.TextField(blank=True, null=True, help_text="JSON data of missing items")
    
    # Gemini comparison results
    gemini_analysis = models.TextField(blank=True, null=True, help_text="JSON data from Gemini room comparison")
    analysis_image = models.ImageField(upload_to='model_validation/comparison/', null=True, blank=True)
    
    # Damage detection results
    damage_analysis = models.TextField(blank=True, null=True, help_text="JSON data from damage detection")
    damage_image = models.ImageField(upload_to='model_validation/damages/', null=True, blank=True)
    
    # For showcasing purposes
    is_showcase = models.BooleanField(default=False, help_text="Whether this validation should be showcased")
    showcase_order = models.IntegerField(default=0, help_text="Order to display in showcase")
    
    def __str__(self):
        return f"{self.get_validation_type_display()} - {self.created_at}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Model Validations"
    
    def get_checkin_objects(self):
        """Get checkin objects as Python object"""
        if not self.checkin_objects:
            return []
        try:
            return json.loads(self.checkin_objects)
        except Exception as e:
            print(f"Error parsing checkin objects: {e}")
            return []
    
    def get_checkout_objects(self):
        """Get checkout objects as Python object"""
        if not self.checkout_objects:
            return []
        try:
            return json.loads(self.checkout_objects)
        except Exception as e:
            print(f"Error parsing checkout objects: {e}")
            return []
    
    def get_missing_items(self):
        """Get missing items as Python object"""
        if not self.missing_items:
            return {}
        try:
            return json.loads(self.missing_items)
        except Exception as e:
            print(f"Error parsing missing items: {e}")
            return {}
    
    def get_gemini_analysis(self):
        """Get Gemini analysis as Python object"""
        if not self.gemini_analysis:
            return {}
        try:
            return json.loads(self.gemini_analysis)
        except Exception as e:
            print(f"Error parsing Gemini analysis: {e}")
            return {}
    
    def get_damage_analysis(self):
        """Get damage analysis as Python object"""
        if not self.damage_analysis:
            return {}
        try:
            return json.loads(self.damage_analysis)
        except Exception as e:
            print(f"Error parsing damage analysis: {e}")
            return {}

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

class MisplacedItemsAnalysis(models.Model):
    """Model to store analysis of misplaced items between check-in and check-out"""
    ANALYSIS_STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    room_activity = models.ForeignKey(RoomActivity, on_delete=models.CASCADE, related_name='misplaced_analyses')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=15, choices=ANALYSIS_STATUS_CHOICES, default='pending')
    
    # Store the processed images
    checkin_image = models.ImageField(upload_to='misplaced_analysis/images/checkin/', null=True, blank=True)
    checkout_image = models.ImageField(upload_to='misplaced_analysis/images/checkout/', null=True, blank=True)
    
    # Visualization (side-by-side comparison)
    visualization = models.ImageField(upload_to='misplaced_analysis/visualizations/', null=True, blank=True)
    
    # Analysis results
    gemini_analysis = models.TextField(blank=True, null=True, help_text="Raw analysis text from Gemini AI")
    structured_analysis = models.TextField(blank=True, null=True, help_text="JSON structured analysis data")
    
    # Cleanliness assessment
    cleanliness_score = models.FloatField(default=0.0, help_text="Overall cleanliness score (0-100)")
    cleanliness_assessment = models.TextField(blank=True, null=True, help_text="Detailed cleanliness assessment")
    
    # Repair assessment
    repair_assessment = models.TextField(blank=True, null=True, help_text="Assessment of repairs needed")
    repair_cost_estimate = models.DecimalField(max_digits=10, decimal_places=2, default=0.00, help_text="Estimated repair costs")
    repair_items = models.TextField(blank=True, null=True, help_text="JSON data of repair items with costs")
    
    def __str__(self):
        return f"Misplaced Items Analysis for Room {self.room_activity.room_number} - {self.created_at.date()}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Misplaced Items Analyses"
    
    def get_structured_analysis(self):
        """Get structured analysis as Python object"""
        if not self.structured_analysis:
            return {}
        try:
            return json.loads(self.structured_analysis)
        except Exception as e:
            print(f"Error parsing structured analysis: {e}")
            return {'error': 'Could not parse structured analysis'}
            
    def get_repair_items(self):
        """Get repair items as Python object"""
        if not self.repair_items:
            return []
        try:
            return json.loads(self.repair_items)
        except Exception as e:
            print(f"Error parsing repair items: {e}")
            return []
