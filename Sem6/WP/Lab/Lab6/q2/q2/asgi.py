"""ASGI config for q2 project."""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "q2.settings")
application = get_asgi_application()
