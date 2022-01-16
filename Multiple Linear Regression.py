import pandas as pd
import keras
from keras import layers

data = pd.read_csv('./dataset/Advertising.csv')
data.head()

x=data[data.columns[1:-1]]
y=data.iloc[:,-1]

model = keras.Sequential()
model.add(layers.Dense(1,input_dim=3))  #y_pred = wl*t1 +w2*t2 +w3*t3 + b
model.summary()

model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=2000)

model.predict(pd.DataFrame[300,0,0])
