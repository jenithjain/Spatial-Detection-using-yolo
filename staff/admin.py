from django.contrib import admin
from .models import StaffMember, RoomActivity, CheckoutAnalysis, ModelValidation, MisplacedItemsAnalysis

# Register your models here.
admin.site.register(StaffMember)
admin.site.register(RoomActivity)
admin.site.register(CheckoutAnalysis)
admin.site.register(ModelValidation)
admin.site.register(MisplacedItemsAnalysis)
