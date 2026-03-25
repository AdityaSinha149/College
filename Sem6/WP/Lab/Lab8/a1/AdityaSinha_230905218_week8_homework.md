# Lab 8 — Grocery Checklist Web Application (Homework)

**Name:** Aditya Sinha  
**Reg. No:** 230905218  
**Class & Section:** CSE-A1  
**Roll No:** 27

---

## Q1 — Grocery Checklist Generation
**Description:**
Develop a web application for grocery checklist generation. On page load, display grocery items as checkboxes. When the user clicks the **Add Item** button, the selected items and their prices are displayed in a table. The table and its cells have styled borders.

**Code (View):**
```python
from django.shortcuts import render

GROCERY_ITEMS = [
  {"name": "Wheat", "price": 40},
  {"name": "Jaggery", "price": 60},
  {"name": "Dal", "price": 80},
]

def grocery_checklist(request):
  selected_items = []
  if request.method == "POST":
    selected = request.POST.getlist("items")
    for item in GROCERY_ITEMS:
      if item["name"] in selected:
        selected_items.append(item)
  return render(request, "homework/grocery.html", {
    "grocery_items": GROCERY_ITEMS,
    "selected_items": selected_items,
  })
```

**Code (URL config):**
```python
from django.urls import path
from . import views

urlpatterns = [
  path("grocery/", views.grocery_checklist, name="grocery_checklist"),
]
```

**Code (Template):**
```html
<form method="post">
  {% csrf_token %}
  <div>Select Item:</div>
  {% for item in grocery_items %}
    <label><input type="checkbox" name="items" value="{{ item.name }}" {% if item.name in selected_items|map(attribute='name') %}checked{% endif %}> {{ item.name }}</label><br>
  {% endfor %}
  <button type="submit">Add Item</button>
</form>

{% if selected_items %}
<table style="border-collapse: collapse; border: 1px solid #888; margin-top: 1em;">
  <tr>
    <th style="border: 1px solid #888; padding: 4px;">Item Name</th>
    <th style="border: 1px solid #888; padding: 4px;">Item Price</th>
  </tr>
  {% for item in selected_items %}
  <tr>
    <td style="border: 1px solid #888; padding: 4px;">{{ item.name }}</td>
    <td style="border: 1px solid #888; padding: 4px;">{{ item.price }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
```

---

## Q2 — (Reserved for additional homework if needed)

---