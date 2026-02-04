from django.urls import path
from .views import send_message

urlpatterns = [
    path("msg/", send_message, name="send_message"),
]
