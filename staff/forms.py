from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import StaffMember, RoomActivity
from manager.models import Manager, Room

class StaffRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(max_length=254, required=True)
    phone_number = forms.CharField(max_length=15, required=True)
    role = forms.CharField(max_length=50, required=True)
    manager_id = forms.IntegerField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def clean_manager_id(self):
        manager_id = self.cleaned_data.get('manager_id')
        try:
            Manager.objects.get(id=manager_id)
            return manager_id
        except Manager.DoesNotExist:
            raise forms.ValidationError("Invalid manager ID. Please enter a valid manager ID.")
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            manager = Manager.objects.get(id=self.cleaned_data['manager_id'])
            staff_member = StaffMember(
                user=user,
                manager=manager,
                phone_number=self.cleaned_data['phone_number'],
                role=self.cleaned_data['role']
            )
            staff_member.save()
        
        return user

class RoomCheckInForm(forms.ModelForm):
    room_id = forms.ModelChoiceField(
        queryset=Room.objects.filter(is_active=True),
        empty_label="Select a room",
        required=True,
        widget=forms.Select(attrs={
            'class': 'form-control',
        }),
        to_field_name='id',
        label="Room"
    )
    notes = forms.CharField(required=False, widget=forms.Textarea(attrs={
        'class': 'form-control',
        'placeholder': 'Enter any notes about the room',
        'rows': 3
    }))
    
    class Meta:
        model = RoomActivity
        fields = ('notes',)
        
    def save(self, commit=True):
        instance = super().save(commit=False)
        room = self.cleaned_data['room_id']
        instance.room_number = room.room_number
        instance.room_id = room.id
        
        if commit:
            instance.save()
        return instance

class RoomCheckOutForm(forms.Form):
    notes = forms.CharField(required=False, widget=forms.Textarea(attrs={
        'class': 'form-control',
        'placeholder': 'Enter any notes about the check-out',
        'rows': 3
    })) 