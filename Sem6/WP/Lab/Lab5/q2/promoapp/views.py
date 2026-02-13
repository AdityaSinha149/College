from django.shortcuts import render


def index(request):
	return render(request, 'q2_promo.html')
