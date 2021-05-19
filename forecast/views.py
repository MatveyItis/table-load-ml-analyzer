from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import csv
from forecast import prediction as pr
from forecast import analyzer_feign as af

pd.options.mode.chained_assignment = None


def forecast(request):
    return render(request, 'forecast.html')


def schema_info(request):
    if request.method == 'GET':
        schema_name = request.GET.get('schema')
        if schema_name is None:
            return HttpResponse(af.get_schemas())
        else:
            return HttpResponse(af.get_tables(schema_name))


def get_test_data(request):
    df = pd.read_csv('data/select_test.csv')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'

    writer = csv.writer(response)
    writer.writerow(['time', 'value'])
    for r in df.values:
        writer.writerow([r[0], r[1]])
    return response


@csrf_exempt
def forecast_data(request):
    if request.method == 'GET':
        schema_name = request.GET.get('schemaName')
        table_name = request.GET.get('tableName')
        query_type = request.GET.get('queryType')
        period = request.GET.get('period')
        pr.start_learning(schema_name, table_name, query_type, period)
        return HttpResponse(status=201)
    return render(request, 'forecast.html')


def get_prediction_result(request):
    if request.method == 'GET':
        schema_name = request.GET.get('schemaName')
        table_name = request.GET.get('tableName')
        query_type = request.GET.get('queryType')
        return pr.read_prediction_result(schema_name, table_name, query_type)
    return render(request, 'forecast.html')


def get_model_configuration(request):
    return render(request, 'model_configuration.html')
