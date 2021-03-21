from django.db import models


class ForecastResults(models.Model):
    schema_name = models.CharField(max_length=40)
    table_name = models.CharField(max_length=40)
    query_type = models.CharField(max_length=40)
    forecast = models.FileField()

    def __str__(self):
        return self.forecast
