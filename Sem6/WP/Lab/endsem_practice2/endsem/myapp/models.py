from django.db import models

class Product(models.Model):
    title = models.CharField(max_length=100)
    price = models.IntegerField()
    description = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.title