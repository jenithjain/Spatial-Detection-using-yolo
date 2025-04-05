from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from .models import StaffMember

class StaffBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            user = UserModel.objects.get(username=username)
            if user.check_password(password):
                try:
                    staff = StaffMember.objects.get(user=user)
                    return user
                except StaffMember.DoesNotExist:
                    return None
        except UserModel.DoesNotExist:
            return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None 