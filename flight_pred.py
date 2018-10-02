# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:34:53 2018

@author: Jayant
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 

weather_data_train = pd.read_csv('flight_predictor_data/weather_data_train.csv')
weather_data_test = pd.read_csv('flight_predictor_data/weather_data_test.csv')
flight_data = pd.read_csv('flight_predictor_data/flight_data_train.csv')

columns = ['Spot'+str(i) + ' totalFlights' for i in range(1,289)]



#Making Testing data
df2=pd.DataFrame()
df2['Day Id']=weather_data_test['Day_Id']
list=[]
list.append('Dew Point')
list.append('Pressure')
list.append('Temperature')
list.append('Wind Speed')
list.append('Wind Direction')
for k in range(0,5):
    for j in range(1,46):
        list1 = []
        for i in range(1,6):
            list1.append('Station'+str(i)+ ' '+list[k]+' Height' + str(j))
        weather_data_test['Total '+list[k]+' Height'+str(j)] = weather_data_test[list1].mean(axis=1)

    list2=[]        
    for i in range(1,46):
        list2.append('Total '+list[k]+' Height'+str(i))
    
    df2['Mean '+list[k]]=weather_data_test[list2].mean(axis=1)
print(df2.head())
#df['total_flights'] = flight_data[columns].sum(axis=1)
df2.to_csv('weather_data_test_compressed.csv', encoding='utf-8', index = False)


df=pd.DataFrame()
df['Day Id']=weather_data_train['Day_Id']

list=[]
list.append('Dew Point')
list.append('Pressure')
list.append('Temperature')
list.append('Wind Speed')
list.append('Wind Direction')

station_list = []
for i in range(1, 6):
    station_list.append(i)
    
for k in range(0,1):
    for j in range(1,46):
        stationdew = pd.DataFrame()
        list1 = []
        for i in range(1,6):
            list1.append('Station'+str(i)+ ' '+list[k]+' Height' + str(j))
            stationdew['Station'+str(i)+' '+list[k]+' Height' +str(j)] = weather_data_train['Station'+str(i)+ ' '+list[k]+' Height' + str(j)]
        plt.plot(station_list,stationdew.iloc[1],label=j)
            
        weather_data_train['Total '+list[k]+' Height'+str(j)] = weather_data_train[list1].mean(axis=1)
    plt.scatter(1,2)
        
    
    list2=[]        
    for i in range(1,46):
        list2.append('Total '+list[k]+' Height'+str(i))
    
    df['Mean '+list[k]]=weather_data_train[list2].mean(axis=1)
print(df.head())
df['total_flights'] = flight_data[columns].sum(axis=1)
df.to_csv('weather_data_train_compressed.csv', encoding='utf-8', index = False)



model = linear_model.LinearRegression()
print (df.shape)
print (df2.shape)
print (df['total_flights'].shape)

#print (weather_data_test())
df.iloc[:,1:6]
model.fit(df.iloc[:,1:6], df['total_flights'])

pred = model.predict(df2.iloc[:,1:6])

#print(weather_data_test.head())
♠


