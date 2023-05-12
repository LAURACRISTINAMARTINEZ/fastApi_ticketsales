import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_transform(input_path:str)-> pd.DataFrame:
    # Carga de datos
    sales = pd.read_csv(input_path,sep='\t', header = None ,encoding= 'utf-8')
    sales.columns =['salesid','listid','sellerid','buyerid','eventid','dateid','qtysold','pricepaid','commission','saletime']
    data = sales[['qtysold', 'saletime']]
    data['saletime'] = pd.to_datetime(data['saletime'], infer_datetime_format=True)
    # Tranformación datos
    data = data.set_index('saletime')
    data = data.sort_index()
    diario = data.resample('D').sum()

    return diario

def transform_dataframe_api(dataframe: pd.DataFrame)-> pd.DataFrame:
    dataframe['saletime'] = pd.to_datetime(dataframe['saletime'], infer_datetime_format=True)
    dataframe = dataframe.set_index('saletime')
    dataframe = dataframe.sort_index()
    diario = dataframe.resample('D').sum()
    return diario

# convertir series en aprendizaje supervisado
def series_to_supervised(data: np.ndarray, n_in:int = 1, n_out:int =1, dropnan: bool = True)-> pd.DataFrame:
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # juntar todo
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # cortar filas con valores NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg