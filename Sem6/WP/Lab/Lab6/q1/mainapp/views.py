from django.shortcuts import render


def index(request):
    result = None
    error = None
    a = ""
    b = ""
    op = "add"

    if request.method == "POST":
        a = request.POST.get("a", "").strip()
        b = request.POST.get("b", "").strip()
        op = request.POST.get("operation", "add")
        try:
            a_val = int(a)
            b_val = int(b)
            if op == "add":
                result = a_val + b_val
            elif op == "sub":
                result = a_val - b_val
            elif op == "mul":
                result = a_val * b_val
            elif op == "div":
                if b_val == 0:
                    error = "Cannot divide by zero."
                else:
                    result = a_val / b_val
        except ValueError:
            error = "Please enter valid integers."

    return render(
        request,
        "q1_index.html",
        {"result": result, "error": error, "a": a, "b": b, "op": op},
    )
