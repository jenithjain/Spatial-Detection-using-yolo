from django.contrib import admin
from .models import Manager, Room, RoomInventory

# Register your models here.
admin.site.register(Manager)
admin.site.register(Room)
admin.site.register(RoomInventory)
