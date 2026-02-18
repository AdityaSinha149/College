from django.shortcuts import render

BOOK = {
    "title": "The Art of Web",
    "author": "Aditya Sinha",
    "year": "2026",
    "publisher": "Campus Press",
    "isbn": "978-1-23456-789-0",
    "cover": "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?auto=format&fit=crop&w=800&q=60",
    "summary": "A concise walk-through of building interactive web applications.",
    "reviews": [
        "Clear and practical guide for students.",
        "Great examples and easy to follow.",
        "Perfect quick reference for labs.",
    ],
}


def index(request):
    page = request.GET.get("page", "home")
    context = {"book": BOOK, "page": page}
    return render(request, "q3_index.html", context)
