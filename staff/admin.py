from django.contrib import admin
from .models import StaffMember, RoomActivity, CheckoutAnalysis

# Register your models here.
admin.site.register(StaffMember)
admin.site.register(RoomActivity)
admin.site.register(CheckoutAnalysis)
