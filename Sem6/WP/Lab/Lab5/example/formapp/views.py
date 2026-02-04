from django.shortcuts import render

# Create your views here.

def send_message(request):
    msg = request.GET.get("msg", "")
    return render(request, "msg.html", {
        "message": msg
    })
