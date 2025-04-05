from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Manager

class ManagerRegistrationForm(UserCreationForm):
    organization_name = forms.CharField(max_length=100)
    contact_number = forms.CharField(max_length=20)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'organization_name', 'contact_number')
        
    def save(self, commit=True):
        user = super().save(commit=False)
        user.save()
        
        manager = Manager.objects.create(
            user=user,
            organization_name=self.cleaned_data.get('organization_name'),
            contact_number=self.cleaned_data.get('contact_number')
        )
        return user 