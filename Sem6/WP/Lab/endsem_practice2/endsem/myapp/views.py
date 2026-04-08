from django.shortcuts import redirect, render
from .models import Product
from .forms import ProductForm

def product(request):
    form = ProductForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("product")
    context = {
        "form": form
    }
    return render(request, "product.html", context)

def view_product(request):
    products = Product.objects.all().order_by("price")
    return render(request, "view_product.html", {"products": products})