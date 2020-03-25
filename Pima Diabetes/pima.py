import tensorflow  as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,MaxPool2D,Conv2D


dataset = pd.read_csv('diabetes.csv')
df = dataset.copy()

Y = dataset['Outcome']
df.drop(['Outcome'],axis = 1,inplace = True)
X = df[:]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

model.summary()

model.fit(X,Y,validation_split=0.10, epochs=150, batch_size=10)














