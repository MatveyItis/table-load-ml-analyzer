import json
import os
import pandas as pd
import csv
from django.http import HttpResponse
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
from multiprocessing import Pool, Process
from . import analyzer_feign as af

pd.options.mode.chained_assignment = None

prefix_cache = 'saved_models/'
forecast_coefficient = 0.3
max_data_len_to_resample_in_minute = 2880


def start_learning(schema, table, query_type):
    # start_date = '2021-03-01 19:41:00.171676'
    csv_file = af.get_csv_file_with_pandas(schema, table, query_type)
    if csv_file is None:
        print(f"Cannot get csv file for schema = %s, table = %s ans query_type = %s", schema, table, query_type)
    else:
        model_prefix = schema + '_' + table + '_' + query_type
        data = csv_file
        p = Process(target=make_prediction, args=(data, model_prefix,))
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


def load_model(file_name):
    with open(prefix_cache + file_name, 'r') as fin:
        return model_from_json(json.load(fin))


def save_model(file_name, model):
    file = prefix_cache + file_name
    if os.path.isfile(file):
        os.remove(file)
    with open(file, 'w') as fout:
        json.dump(model_to_json(model), fout)


def save_prediction_result(model_name, arr, predict):
    predict['value'] = '0'
    i = 0
    pr_val_col = predict['value']
    for d in arr:
        pr_val_col[i] = d
        i += 1
    predict.to_csv('prediction_result/' + model_name + '.csv', index=False)


# method to read prediction values from csv files
def read_prediction_result(model_file_prefix):
    model_file_prefix = model_file_prefix + '.csv'
    full_path = 'prediction_result/' + model_file_prefix
    if os.path.isfile(full_path):
        data = pd.read_csv(full_path, index_col=['ds'], parse_dates=['ds'])
        return HttpResponse(resample_data(data).to_csv())
    else:
        return HttpResponse(status=404)


def prepare_data(arr, predict):
    predict['value'] = '0'
    i = 0
    pr_val_col = predict['value']
    for d in arr:
        pr_val_col[i] = d
        i += 1
    # predict = resample_data(predict)
    return predict.to_csv(index=False)


def make_prediction(data=None, model_file_prefix='', schema='', table='', query_type=''):
    model_file_name = model_file_prefix + '_model.json'
    print('Started prediction for model = ', model_file_name)
    is_cache_existed = os.path.isfile(model_file_name)
    if is_cache_existed:
        m = load_model(model_file_name)
        data = m.history
        last_date = max(data.ds)
        print(last_date)
        add_data = af.get_csv_file_with_pandas(schema, table, query_type)
        if add_data is None:
            return m.history
        add_data.columns = ['ds', 'y']
        add_data = add_data.loc[pd.to_datetime(add_data['ds']) > last_date, :]
        if add_data.size < 10:
            return m.history
        else:
            new_m = Prophet(interval_width=0.95, daily_seasonality=True)
            # todo thinking about new data appending and visualization
            new_m.fit(data.append(add_data), init=stan_init(m))
            prediction_periods = int(len(data) * forecast_coefficient)
            future = new_m.make_future_dataframe(periods=prediction_periods, freq='T')
            predict = new_m.predict(future)
            new_m.fit_kwargs['init']['delta'] = new_m.fit_kwargs['init']['delta'].tolist()
            new_m.fit_kwargs['init']['beta'] = new_m.fit_kwargs['init']['beta'].tolist()
            save_model(model_file_name, new_m)
            save_prediction_result(model_file_prefix, data.y, predict)
            return prepare_data(data.y, predict)
    else:
        prediction_periods = int(len(data) * forecast_coefficient)
        data.columns = ['ds', 'y']
        m = Prophet(interval_width=0.95, daily_seasonality=True)
        m.fit(data)
        future = m.make_future_dataframe(periods=prediction_periods, freq='T')
        predict = m.predict(future)
        save_model(model_file_name, m)
        save_prediction_result(model_file_prefix, data['y'], predict)
        return prepare_data(data['y'], predict)


def resample_data(df):
    if len(df) > max_data_len_to_resample_in_minute:
        return df.resample('30T').sum()
    else:
        return df
