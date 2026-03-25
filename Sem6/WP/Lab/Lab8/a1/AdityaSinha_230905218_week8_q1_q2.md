# Lab 8 — Q1 and Q2 Django Apps

**Name:** Aditya Sinha  
**Reg. No:** 230905218  
**Class & Section:** CSE-A1  
**Roll No:** 27

---

## Q1 — Car Manufacturer & Model Selection
**Description:**
A two-page Django app to select a car manufacturer and model, then display the selection.

**Code (View):**
```python
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
```

**Code (URL config):**
```python
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("result/", views.result, name="result"),
]
```

**Code (Templates):**
`index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Select Car</title>
    <link rel="stylesheet" href="/static/q1.css" />
</head>
<body>
  <div class="card">
    <h1>Select Car Manufacturer & Model</h1>
    <form method="post">
      {% csrf_token %}
      <label>Manufacturer</label>
      <select name="manufacturer">
        {% for m in manufacturers %}
          <option value="{{ m }}">{{ m }}</option>
        {% endfor %}
      </select>

      <label>Model Name</label>
      <input type="text" name="model" />

      <button type="submit">Submit</button>
    </form>
  </div>
</body>
</html>
```

`result.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Selected Car</title>
    <link rel="stylesheet" href="/static/q1.css" />
</head>
<body>
  <div class="card">
    <h1>Selected Car</h1>
    <p><strong>Manufacturer:</strong> {{ manufacturer }}</p>
    <p><strong>Model:</strong> {{ model }}</p>
    <a href="/">Back</a>
  </div>
</body>
</html>
```

---

## Q2 — Student Details Form (Two Page)
**Description:**
A two-page Django app to enter student details and subject, then display the data on a second page. Data is stored in the session.

**Code (View):**
```python
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
```

**Code (URL config):**
```python
from django.urls import path
from . import views

urlpatterns = [
    path("", views.first_page, name="first_page"),
    path("second/", views.second_page, name="second_page"),
]
```

**Code (Templates):**
`firstPage.html`
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>First Page</title>
  <link rel="stylesheet" href="/static/q2.css">
</head>
<body>
  <div class="card">
    <h1>Enter Details</h1>
    <form method="post">
      {% csrf_token %}
      <label>Name</label>
      <input type="text" name="name" value="{{ form_data.name }}">

      <label>Roll</label>
      <input type="text" name="roll" value="{{ form_data.roll }}">

      <label>Subject</label>
      <select name="subject">
        {% for s in subjects %}
          <option value="{{ s }}" {% if s == form_data.subject %}selected{% endif %}>{{ s }}</option>
        {% endfor %}
      </select>

      <button type="submit">Submit</button>
    </form>
  </div>
</body>
</html>
```

`secondPage.html`
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Second Page</title>
  <link rel="stylesheet" href="/static/q2.css">
</head>
<body>
  <div class="card">
    <h1>Submitted Data</h1>
    <p><strong>Name:</strong> {{ data.name }}</p>
    <p><strong>Roll:</strong> {{ data.roll }}</p>
    <p><strong>Subject:</strong> {{ data.subject }}</p>

    <form method="post">
      {% csrf_token %}
      <button type="submit">Back to First Page</button>
    </form>
  </div>
</body>
</html>
```

---
