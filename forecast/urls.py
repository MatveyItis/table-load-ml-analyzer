from django.urls import path
from . import views
from django.conf.urls import url
from rest_framework_swagger.views import get_swagger_view

schema_view = get_swagger_view(title='Pastebin API')

app_name = 'forecast'

urlpatterns = [
    path('', views.forecast, name='forecast'),
    path('forecast', views.forecast_data, name='forecast_data'),
    path('forecast/result', views.get_prediction_result, name='prediction_result'),
    path('database/schema', views.schema_info, name='schema_info'),
    path('test/data', views.get_test_data, name='test_data'),
    path('model/configuration', views.get_model_configuration, name='model_configuration'),
    url(r'^swagger$', schema_view)
]
