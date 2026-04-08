from django.db.models import Avg
from django.shortcuts import redirect, render

from .forms import SuggestionForm
from .models import Suggesstion


def home(request):
    return render(request, "home.html")


def suggestions(request):
    form = SuggestionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("suggestions")

    return render(request, "suggestions.html", {"form": form})


def view_suggestions(request):
    data = Suggesstion.objects.all().order_by("-id")
    average_rating = data.aggregate(avg=Avg("rating"))["avg"] or 0
    context = {
        "suggestions": data,
        "average_rating": round(average_rating, 2),
    }
    return render(request, "viewsuggestions.html", context)

