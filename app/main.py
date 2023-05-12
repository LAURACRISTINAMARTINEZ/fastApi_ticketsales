import logging
import uvicorn
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from utils import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from typing import List, Optional, Union, Tuple, Any

logging.getLogger("tensorflow").setLevel(logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = Path(__file__).parent / 'model'



def agregarTensor(tensor:np.ndarray, nuevotensor:np.ndarray)-> np.ndarray:

    for i in range(tensor.shape[2]-1):
        tensor[0][0][i] = tensor[0][0][i+1]
    tensor[0][0][tensor.shape[2]-1]=nuevotensor

    return tensor

class LoadedModel:
    def __init__(self, model: tf.keras.models.Sequential, Pasos: int):
        self.model = model
        self.Pasos = Pasos


    def make_prediction(self, tensor):
        results=[]
        for i in range(7):
            parcial= self.model.predict(tensor)
            results.append(parcial[0])
            tensor=agregarTensor(tensor,parcial[0])
        return results

    
def load_model_components(model_path:str) -> Tuple[Any]:
        try:
            model = load_model(str(model_path))

            with open(model_path/'model_config.yaml', 'r') as f:
                model_config = yaml.safe_load(f)

        except (FileNotFoundError, OSError) as e:
            model = None
        
        return model, model_config


def get_tensor(datos: pd.DataFrame, model_conf):

    try:
        diariosales = preprocessing.transform_dataframe_api(datos)
        # Preparación de los datos
        values = diariosales.values
        # todos los datos sean flotantes
        values = values.astype('float32')
        # normalizacion
        scaler = MinMaxScaler(feature_range=(-1, 1))
        values=values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        # aprendizaje supervisado
        reframed = preprocessing.series_to_supervised(scaled, model_conf['pasos'], 1)
        reframed.drop(reframed.columns[[model_conf['pasos']]], axis=1, inplace=True)
        values = reframed.values
        tensor = values[4:, :]
        tensor = tensor.reshape((tensor.shape[0], 1, tensor.shape[1]))
    
    except Exception as e:
        print(e)
        tensor = None
    
    return tensor, scaler


model, model_config = load_model_components(model_path=model_path)
model_time = LoadedModel(model=model, Pasos= model_config['pasos'])

#api en fastapi
app = FastAPI(title='APP PREDICT FUTURE SALES')

class Historico(BaseModel):
    saletime: List[str] = []
    qtysold: List[int] = []

@app.get('/')

def root():
    html_content = """
    <html>
        <meta http-equiv=”Content-Type” content=”text/html; charset=UTF-8″ />
        <head>
            <title>Servicio</title>
        </head>
        <body>
            <h1>App para la predicción de ventas de boleteria</h1>
            <h2>LCMC</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)



@app.post("/predict/")
async def predict_from_input(input: Historico):

    if input is not None:
        answer = {'prediction':{}}

        data = input.dict()
        data = pd.DataFrame.from_dict(data)
        tensor, scaler = get_tensor(datos=data, model_conf=model_config )

        prediction = model_time.make_prediction(tensor)
        adimen = [x for x in prediction]    
        inverted = scaler.inverse_transform(adimen)

        for ans in range(len(inverted)):
            answer['prediction'][f'dia{ans+1}'] = int(inverted[ans])
        
        return answer

    else:
        return {'answer': 'Model not found'}


if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port=5010)






