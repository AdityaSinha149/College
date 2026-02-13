from django.shortcuts import render


def index(request):
	return render(request, 'a1_captcha.html')
