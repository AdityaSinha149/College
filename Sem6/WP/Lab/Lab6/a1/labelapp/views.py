from django.shortcuts import render


def index(request):
    name = ""
    message = ""
    bold = False
    italic = False
    underline = False
    color = "red"
    label_text = ""
    exited = False

    if request.method == "POST":
        action = request.POST.get("action", "display")
        name = request.POST.get("name", "").strip()
        message = request.POST.get("message", "").strip()
        bold = request.POST.get("bold") == "on"
        italic = request.POST.get("italic") == "on"
        underline = request.POST.get("underline") == "on"
        color = request.POST.get("color", "red")

        if action == "display":
            label_text = f"Name:{name} Message:{message}".strip()
        elif action == "clear":
            name = ""
            message = ""
            label_text = ""
            bold = False
            italic = False
            underline = False
            color = "red"
        elif action == "exit":
            exited = True
            name = ""
            message = ""
            label_text = ""
            bold = False
            italic = False
            underline = False
            color = "red"

    context = {
        "name": name,
        "message": message,
        "bold": bold,
        "italic": italic,
        "underline": underline,
        "color": color,
        "label_text": label_text,
        "exited": exited,
    }
    return render(request, "a1_label.html", context)
