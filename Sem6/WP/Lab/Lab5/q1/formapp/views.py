from django.shortcuts import render


def index(request):
	return render(request, 'q1_form.html')
