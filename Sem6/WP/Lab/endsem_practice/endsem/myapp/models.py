from django.db import models


class Suggesstion(models.Model):
    TYPE_CHOICES = [
        ("trainer", "Trainer"),
        ("student", "Student"),
        ("staff", "Staff"),
    ]

    name = models.CharField(max_length=100)
    email = models.EmailField()
    types = models.CharField(max_length=20, choices=TYPE_CHOICES)
    rating = models.IntegerField()

    def __str__(self):
        return f"{self.name} ({self.rating})"