import os
import pandas as pd
import numpy as np
import time
import redis
from django.http import HttpResponse
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from multiprocessing import Process
from . import analyzer_feign as af

pd.options.mode.chained_assignment = None

prefix_cache = 'ml-cache-'
max_data_len_to_resample_in_minute = 2880


def start_learning(schema, table, query_type, start_date, period):
    print(f"called start_learning method with parameters: schema = %s, table = %s, query_type = %s, "
          "start_date = %s and period = %s" % (schema, table, query_type, start_date, period))
    csv_file = af.get_csv_file_with_pandas(schema, table, query_type, start_date)
    if csv_file is None:
        print(f"Cannot get csv file for schema = %s, table = %s, query_type = %s, start_date = %s",
              schema, table, query_type, start_date)
    else:
        model_prefix = schema + '_' + table + '_' + query_type + '_' + start_date
        data = csv_file
        p = Process(target=make_prediction, args=(data, model_prefix, period))
        p.start()


def stan_init(m):
    print("called stan_init method")
    res = {}
    print(m.params)
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        try:
            res[pname] = m.params[pname][0]
        except RuntimeError:
            print("Some parameters were not found")
            print("result params = ", res)
        finally:
            return res


def mape(y, y_pred):
    return (1 - np.mean(np.abs((y - y_pred) / y))) * 100


def load_model(file_name, r):
    print(f"called load_model method with parameters: file_name = %s" % file_name)
    key = prefix_cache + file_name
    return model_from_json(r.get(key))


def save_model(file_name, model, r):
    print(f"called save_model method with parameters: file_name = %s" % file_name)
    file = prefix_cache + file_name
    r.mset({file: model_to_json(model)})


def save_prediction_result(model_name, arr, predict, mape):
    print(f"called save_prediction_result method for model = %s" % model_name)
    predict['value'] = '0'
    pr_val_col = predict['value']
    predict['mape'] = mape
    i = 0
    for d in arr:
        pr_val_col[i] = d
        i += 1
    predict.to_csv('prediction_result/' + model_name + '.csv', index=False)


# method to read prediction values from csv files
def read_prediction_result(schema, table, query_type, start_date):
    print("called read_prediction_result method with parameters: schema = %s, "
          "table = %s, query_type = %s, start_date = %s" % (schema, table, query_type, start_date))
    model_file_prefix = schema + '_' + table + '_' + query_type + '_' + start_date + '.csv'
    full_path = 'prediction_result/' + model_file_prefix
    if os.path.isfile(full_path):
        data = pd.read_csv(full_path, index_col=['ds'], parse_dates=['ds'])
        return HttpResponse(data.to_csv(), content_type="text/plain;charset=UTF-8")
    else:
        data = af.get_csv_file_with_pandas(schema, table, query_type, start_date)
        data.columns = ['ds', 'value']
        data = data.set_index(data['ds'])
        data.index = pd.to_datetime(data.index)
        return HttpResponse(data.to_csv(), content_type="text/plain;charset=UTF-8")


def prepare_data(arr, predict):
    print("called prepare_data method")
    predict['value'] = '0'
    i = 0
    pr_val_col = predict['value']
    for d in arr:
        pr_val_col[i] = d
        i += 1
    return predict.to_csv(index=False)


def make_prediction(data=None, model_file_prefix='', period='', schema='', table='', query_type=''):
    print("called make_prediction method for model = %s and period = %s" % (model_file_prefix, period))
    start_time = time.time()

    model_file_name = model_file_prefix + '_model.json'
    r = redis.Redis()
    is_cache_existed = r.exists(prefix_cache + model_file_name)
    if is_cache_existed:
        loaded_m = load_model(model_file_name, r)

        data.columns = ['ds', 'y']
        test_length = int(len(data) * 0.1)
        df_train = data.iloc[:-test_length]
        print(f'train set length = {len(df_train)}')
        df_test = data.iloc[-test_length:]
        print(f'test set length = {len(df_test)}')
        m = create_prophet_model()
        m.fit(df_train, init=stan_init(loaded_m))
        future = m.make_future_dataframe(periods=test_length, freq='T')
        predict = m.predict(future)
        df_test['actual'] = predict.iloc[-test_length:].loc[:, 'yhat']
        df_test.set_index('ds', inplace=True)
        prophet_mape = mape(df_test['y'], df_test['actual'])
        print("Model mape = ", prophet_mape)

        new_m = create_prophet_model()
        new_m.fit(data, init=stan_init(m))
        prediction_periods = resolve_prediction_periods(period)
        result_future = new_m.make_future_dataframe(periods=prediction_periods, freq='T')
        result_predict = new_m.predict(result_future)
        new_m.fit_kwargs['init']['delta'] = new_m.fit_kwargs['init']['delta'].tolist()
        try:
            new_m.fit_kwargs['init']['beta'] = new_m.fit_kwargs['init']['beta'].tolist()
        except KeyError:
            print("beta key not found")
        save_model(model_file_name, new_m, r)
        save_prediction_result(model_file_prefix, data.y, result_predict, prophet_mape)

        end_time = time.time()
        print('Model with stan init learning time(ms) = ', (end_time - start_time) * 1000)

        return prepare_data(data.y, predict)
    else:
        prediction_periods = resolve_prediction_periods(period)
        data.columns = ['ds', 'y']
        test_length = int(len(data) * 0.1)
        df_train = data.iloc[:-test_length]
        print(f'train set length = {len(df_train)}')
        df_test = data.iloc[-test_length:]
        print(f'test set length = {len(df_test)}')
        m = create_prophet_model()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=test_length, freq='T')
        predict = m.predict(future)
        df_test['actual'] = predict.iloc[-test_length:].loc[:, 'yhat']
        df_test.set_index('ds', inplace=True)
        prophet_mape = mape(df_test['y'], df_test['actual'])
        print("Model mape = ", prophet_mape)
        result_m = create_prophet_model()

        result_m.fit(data, init=stan_init(m))
        result_future = result_m.make_future_dataframe(periods=prediction_periods, freq='T')
        result_predict = result_m.predict(result_future)
        save_model(model_file_name, result_m, r)
        save_prediction_result(model_file_prefix, data['y'], result_predict, prophet_mape)

        end_time = time.time()
        print('Model learning time(ms) = ', (end_time - start_time) * 1000)

        return prepare_data(data['y'], predict)


def create_prophet_model():
    print("called create_prophet_model method")
    return Prophet(interval_width=0.9)


def resample_data(df):
    if len(df) > max_data_len_to_resample_in_minute:
        return df.resample('1H').sum()
    else:
        return df


def resolve_prediction_periods(request_period):
    print(f"called resolve_prediction_periods method with parameters: request_period = %s" % (request_period))
    if 'h' in request_period:
        per = int(request_period.replace('h', ''))
        return 60 * per
    if 'd' in request_period:
        per = int(request_period.replace('d', ''))
        return 24 * 60 * per
