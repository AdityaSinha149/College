from django.urls import include, path
from django.contrib import admin

urlpatterns = [
    path("", include("mainapp.urls")),
    path("admin/", admin.site.urls),
]
