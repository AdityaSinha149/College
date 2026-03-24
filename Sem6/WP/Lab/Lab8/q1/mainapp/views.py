from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode


def index(request):
    """First page: select manufacturer and enter model name."""
    manufacturers = ["Toyota", "Honda", "Ford", "Hyundai", "BMW"]

    if request.method == "POST":
        manufacturer = request.POST.get("manufacturer", "").strip()
        model = request.POST.get("model", "").strip()
        # redirect to result page with query params
        params = urlencode({"manufacturer": manufacturer, "model": model})
        return redirect(f"{reverse('result')}?{params}")

    return render(request, "index.html", {"manufacturers": manufacturers})


def result(request):
    """Result page: display selected manufacturer and model."""
    manufacturer = request.GET.get("manufacturer", "")
    model = request.GET.get("model", "")
    return render(request, "result.html", {"manufacturer": manufacturer, "model": model})
