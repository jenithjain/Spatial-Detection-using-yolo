from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from .models import Manager

class ManagerBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            user = UserModel.objects.get(username=username)
            if user.check_password(password):
                try:
                    manager = Manager.objects.get(user=user)
                    return user
                except Manager.DoesNotExist:
                    return None
        except UserModel.DoesNotExist:
            return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None 