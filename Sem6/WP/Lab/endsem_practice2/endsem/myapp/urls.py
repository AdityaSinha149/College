from django.urls import path
from . import views

urlpatterns = [
    path("product/", views.product, name="product"),
    path("view_product/", views.view_product, name="view_product"),
]