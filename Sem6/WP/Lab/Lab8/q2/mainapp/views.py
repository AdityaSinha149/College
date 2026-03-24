from django.shortcuts import render, redirect


def first_page(request):
    subjects = ["Math", "Physics", "Chemistry", "Computer"]
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        roll = request.POST.get("roll", "").strip()
        subject = request.POST.get("subject", "")
        # store in session
        request.session["form_data"] = {"name": name, "roll": roll, "subject": subject}
        return redirect("second_page")

    # GET
    form_data = request.session.get("form_data") or {"name": "", "roll": "", "subject": subjects[0]}
    return render(request, "firstPage.html", {"subjects": subjects, "form_data": form_data})


def second_page(request):
    data = request.session.get("form_data")
    if not data:
        return redirect("first_page")

    if request.method == "POST":
        # go back to first page (retain session so fields stay populated)
        return redirect("first_page")

    return render(request, "secondPage.html", {"data": data})
