from django.shortcuts import render

DEFAULT_IMAGE = "https://images.unsplash.com/photo-1509021436665-8f07dbf5bf1d?auto=format&fit=crop&w=800&q=60"
ALT_IMAGE_1 = "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=800&q=60"
ALT_IMAGE_2 = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=800&q=60"


def index(request):
    title = "Magazine"
    subtitle = "Your monthly dose"
    tagline = "Design the cover live"
    cover_image = DEFAULT_IMAGE
    bg_color = "#f4f4f4"
    text_color = "#111111"
    font_size = "20"
    font_family = "'Segoe UI', Arial, sans-serif"

    if request.method == "POST":
        title = request.POST.get("title", title)
        subtitle = request.POST.get("subtitle", subtitle)
        tagline = request.POST.get("tagline", tagline)
        cover_image = request.POST.get("cover_image", cover_image)
        bg_color = request.POST.get("bg_color", bg_color)
        text_color = request.POST.get("text_color", text_color)
        font_size = request.POST.get("font_size", font_size)
        font_family = request.POST.get("font_family", font_family)

    context = {
        "title": title,
        "subtitle": subtitle,
        "tagline": tagline,
        "cover_image": cover_image,
        "bg_color": bg_color,
        "text_color": text_color,
        "font_size": font_size,
        "font_family": font_family,
        "images": [DEFAULT_IMAGE, ALT_IMAGE_1, ALT_IMAGE_2],
    }
    return render(request, "q2_index.html", context)
