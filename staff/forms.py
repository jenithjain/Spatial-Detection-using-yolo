from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import StaffMember
from manager.models import Manager

class StaffRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)
    phone_number = forms.CharField(max_length=15, required=True)
    role = forms.CharField(max_length=50, required=True)
    manager = forms.ModelChoiceField(
        queryset=Manager.objects.all(),
        required=True,
        label='Select Organization'
    )

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            StaffMember.objects.create(
                user=user,
                phone_number=self.cleaned_data['phone_number'],
                role=self.cleaned_data['role'],
                manager=self.cleaned_data['manager']
            )
        return user 