from django import forms

from .models import Suggesstion


class SuggestionForm(forms.ModelForm):
    class Meta:
        model = Suggesstion
        fields = ["name", "email", "types", "rating"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Enter your name"}),
            "email": forms.EmailInput(attrs={"placeholder": "Enter your email"}),
            "types": forms.Select(),
            "rating": forms.NumberInput(attrs={"min": 1, "max": 5, "placeholder": "1-5"}),
        }

    def clean_rating(self):
        rating = self.cleaned_data["rating"]
        if rating < 1 or rating > 5:
            raise forms.ValidationError("Rating must be between 1 and 5.")
        return rating
