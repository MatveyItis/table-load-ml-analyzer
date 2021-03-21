import http.client
import pandas as pd

STAT_BACKEND_URL = 'localhost:2789'


def get_file_with_stat(schema, table, query_type):
    try:
        connection = http.client.HTTPConnection(STAT_BACKEND_URL)
        headers = {'Content-type': 'text/csv;charset=UTF-8'}
        url = '/statistics/file?schema=' + schema + '&table=' + table + '&queryType=' + query_type + '&fileType=CSV'
        connection.request(method='GET', url=url, headers=headers)
        response = connection.getresponse()
        decoded_response = response.read().decode()
        print(decoded_response)
        return decoded_response
    except ConnectionError:
        return None


def get_csv_file_with_pandas(schema, table, query_type):
    try:
        url = 'http://' + STAT_BACKEND_URL + '/statistics/file?schema=' + schema + '&table=' + table + '&queryType=' + query_type + '&fileType=CSV'
        return pd.read_csv(url)
    except Exception:
        return None


def get_statistics_info():
    connection = http.client.HTTPConnection(STAT_BACKEND_URL)
    headers = {'Content-type': 'application/json'}
    connection.request(method='GET', url='/database/info', headers=headers)
    response = connection.getresponse()
    decoded_response = response.read().decode()
    return decoded_response


def get_schemas():
    connection = http.client.HTTPConnection(STAT_BACKEND_URL)
    headers = {'Content-type': 'application/json'}
    connection.request(method='GET', url='/database/schema', headers=headers)
    response = connection.getresponse()
    decoded_response = response.read().decode()
    return decoded_response


def get_tables(schema):
    connection = http.client.HTTPConnection(STAT_BACKEND_URL)
    headers = {'Content-type': 'application/json'}
    connection.request(method='GET', url='/database/schema/' + schema, headers=headers)
    response = connection.getresponse()
    decoded_response = response.read().decode()
    return decoded_response
