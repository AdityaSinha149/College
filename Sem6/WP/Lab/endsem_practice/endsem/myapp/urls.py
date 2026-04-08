from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("suggestions/", views.suggestions, name="suggestions"),
    path("viewsuggestions/", views.view_suggestions, name="view_suggestions"),
]