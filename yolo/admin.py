from django.contrib import admin
from .models import Detection

# Register your models here.
@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'detection_type', 'session_id', 'timestamp')
    list_filter = ('detection_type',)
    search_fields = ('session_id',)
    readonly_fields = ('timestamp',)
