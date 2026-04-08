from django import forms
from .models import Product

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['title', 'price', 'description']
        widgets = {
            "title": forms.TextInput(attrs={"placeholder":"Enter Title"}),
            "price": forms.NumberInput(attrs={"placeholder":"Enter Price"}),
            "description": forms.TextInput(attrs={'placeholder':'Enter Description'}),
        }