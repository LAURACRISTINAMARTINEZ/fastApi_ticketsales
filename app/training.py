# %%
# Tratamiento de datos
# ==============================================================================
import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from utils import preprocessing, plotting
# ==============================================================================
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Dropout, LSTM, Bidirectional,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='[%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# %%
OUT_DIR = Path(__file__).parent / 'model'
if not OUT_DIR.exists():
    Path.mkdir(OUT_DIR, parents=True)
PASOS = 5
EPOCHS = 100
DROPOUT = 0.2
LR = 0.0008
model_config = {
    'pasos': PASOS,
    'epochs': EPOCHS,
    'dropout': DROPOUT,
    'batch_size': PASOS,
    'learning_rate': LR,
}
with open(OUT_DIR/'model_config.yaml','w') as f:
    yaml.dump(model_config,f)
# %%
# Carga y transformación de datos
pathdir = "dataset/sales_tab.txt"
diariosales = preprocessing.load_transform(pathdir)
# %%
# Preparación de los datos
values = diariosales.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1)
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = preprocessing.series_to_supervised(scaled, PASOS, 1)
# %%
# Split data
# ============================================================================================
values = reframed.values
n_train_days = diariosales.shape[0] - (30+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
logger.info(f"\nTrain size: {len(train)}")
logger.info(f"Test size: {len(test)}")
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
logger.info(f"x_train shape: {x_train.shape}")
logger.info(f"x_val shape: {x_val.shape}")

# %%
# Build the model
model = Sequential()
model.add(Dense(units =PASOS, activation='relu', input_shape = (1,PASOS)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next value # Predicción del próximo valor
# %%
print(model.summary())
model.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=LR),
              metrics=["mse"])
# %%
# save the best model
mc = ModelCheckpoint(OUT_DIR, monitor='val_loss', save_best_only=True)
# %%
history = model.fit(x_train, y_train,
                    batch_size=PASOS,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[mc],
                    )
# %%
# Plotting training
plotting.plot_training(history, os.path.join(OUT_DIR, 'plot_training.png'))
# %%
# load model
del model
model = load_model(OUT_DIR)
plotting.plot_val(x_val, y_val, model,os.path.join(OUT_DIR, 'plot_val.png'))
# %%
