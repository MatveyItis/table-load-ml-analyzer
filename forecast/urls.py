from django.urls import path
from . import views

app_name = 'forecast'

urlpatterns = [
    path('', views.forecast, name='forecast'),
    path('forecast', views.forecast_data, name='forecast_data'),
    path('forecast/result', views.get_prediction_result, name='prediction_result'),
    path('statistics/info', views.statistics_info, name='statistics_info'),
    path('database/schema', views.schema_info, name='schema_info'),
    path('test/data', views.get_test_data, name='test_data')
]
