import os
import pandas as pd
import time
import redis
from django.http import HttpResponse
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
from multiprocessing import Process
from . import analyzer_feign as af

pd.options.mode.chained_assignment = None

prefix_cache = 'ml-cache-'
forecast_coefficient = 0.3
max_data_len_to_resample_in_minute = 2880


def start_learning(schema, table, query_type, period):
    # start_date = '2021-03-01 19:41:00.171676'
    csv_file = af.get_csv_file_with_pandas(schema, table, query_type)
    if csv_file is None:
        print(f"Cannot get csv file for schema = %s, table = %s ans query_type = %s", schema, table, query_type)
    else:
        model_prefix = schema + '_' + table + '_' + query_type
        data = csv_file
        p = Process(target=make_prediction, args=(data, model_prefix, period))
        p.start()


def stan_init(m):
    """Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.

    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res


def load_model(file_name, r):
    key = prefix_cache + file_name
    return model_from_json(r.get(key))


def save_model(file_name, model, r):
    file = prefix_cache + file_name
    r.mset({file: model_to_json(model)})


def save_prediction_result(model_name, arr, predict):
    predict['value'] = '0'
    i = 0
    pr_val_col = predict['value']
    for d in arr:
        pr_val_col[i] = d
        i += 1
    predict.to_csv('prediction_result/' + model_name + '.csv', index=False)


# method to read prediction values from csv files
def read_prediction_result(schema, table, query_type):
    model_file_prefix = schema + '_' + table + '_' + query_type + '.csv'
    full_path = 'prediction_result/' + model_file_prefix
    if os.path.isfile(full_path):
        data = pd.read_csv(full_path, index_col=['ds'], parse_dates=['ds'])
        return HttpResponse(data.to_csv(), content_type="text/plain;charset=UTF-8")
    else:
        data = af.get_csv_file_with_pandas(schema, table, query_type)
        data.columns = ['ds', 'value']
        data = data.set_index(data['ds'])
        data.index = pd.to_datetime(data.index)
        return HttpResponse(data.to_csv(), content_type="text/plain;charset=UTF-8")


def prepare_data(arr, predict):
    predict['value'] = '0'
    i = 0
    pr_val_col = predict['value']
    for d in arr:
        pr_val_col[i] = d
        i += 1
    # predict = predict.set_index(predict['ds'])
    # predict.index = pd.to_datetime(predict.index)
    # predict = resample_data(predict)
    return predict.to_csv(index=False)


def make_prediction(data=None, model_file_prefix='', period='', schema='', table='', query_type=''):
    start_time = time.time()

    model_file_name = model_file_prefix + '_model.json'
    print('Started prediction for model = ', model_file_name)
    r = redis.Redis()
    is_cache_existed = r.exists(prefix_cache + model_file_name)
    if is_cache_existed:
        m = load_model(model_file_name, r)
        data.columns = ['ds', 'y']
        new_m = create_prophet_model()
        new_m.fit(data, init=stan_init(m))
        prediction_periods = resolve_prediction_periods(period)
        future = new_m.make_future_dataframe(periods=prediction_periods, freq='T')
        predict = new_m.predict(future)
        new_m.fit_kwargs['init']['delta'] = new_m.fit_kwargs['init']['delta'].tolist()
        new_m.fit_kwargs['init']['beta'] = new_m.fit_kwargs['init']['beta'].tolist()
        save_model(model_file_name, new_m, r)
        save_prediction_result(model_file_prefix, data.y, predict)

        end_time = time.time()
        print('Model with stan init learning time(ms) = ', (end_time - start_time) * 1000)

        return prepare_data(data.y, predict)
    else:
        prediction_periods = resolve_prediction_periods(period)
        data.columns = ['ds', 'y']
        m = create_prophet_model()
        m.fit(data)
        future = m.make_future_dataframe(periods=prediction_periods, freq='T')
        predict = m.predict(future)
        save_model(model_file_name, m, r)
        save_prediction_result(model_file_prefix, data['y'], predict)

        end_time = time.time()
        print('Model learning time(ms) = ', (end_time - start_time) * 1000)

        return prepare_data(data['y'], predict)


def create_prophet_model():
    return Prophet(interval_width=0.9)


def resample_data(df):
    if len(df) > max_data_len_to_resample_in_minute:
        return df.resample('1H').sum()
    else:
        return df


def resolve_prediction_periods(request_period):
    if 'h' in request_period:
        per = int(request_period.replace('h', ''))
        return 60 * per
    if 'd' in request_period:
        per = int(request_period.replace('d', ''))
        return 24 * 60 * per
