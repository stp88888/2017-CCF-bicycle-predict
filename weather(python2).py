# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:26:13 2017

@author: STP
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib2
import re

#输入年月(不支持日)，注意时间不包括end_time，例如start_time=201501，end_time=201608，那么2016年8月的数据不会被采集，要令end_time=201609，才能采集2016年8月的数据
#start_time = input('start time:')
#end_time = input('end time:')
start_time = '201501'
end_time = '201511'
start_time_timestamp = pd.to_datetime(start_time, format = '%Y%m')
end_time_timestamp = pd.to_datetime(end_time, format = '%Y%m')

#城市
CITY_NAME = 'yancheng'
#获取天气数据
time = pd.date_range(start_time_timestamp, periods=((end_time_timestamp - start_time_timestamp).days))
final = pd.DataFrame(np.zeros((((end_time_timestamp - start_time_timestamp).days), 6), dtype=float), columns=['day', 'tem_max', 'tem_min', 'weather', 'wind', 'wind_level'])
final['day'] = time
all_merge = []
for vYear in range(start_time_timestamp.year, end_time_timestamp.year + 1):
    if vYear == start_time_timestamp.year:
        Monthrange = np.arange(start_time_timestamp.month, 13)
    elif vYear == end_time_timestamp.year:
        Monthrange = np.arange(1, end_time_timestamp.month)
    else:
        Monthrange = np.arange(1, 13)
    for vMonth in Monthrange:
        for vDay in range(1, 32):
            if vYear % 4 == 0:
                if vMonth == 2 and vDay > 29:
                    break
            else:
                if vMonth == 2 and vDay > 28:
                    break
            if vMonth in [4, 6, 9, 11] and vDay > 30:
                break
        weather = []
        theDate2 = str(vYear) + str(vMonth).zfill(2) + str(vDay).zfill(2)
        print(theDate2)
        #theport = value['AIRPORT_CODE']

        theurl = 'http://zhenjiang.tianqi.com/' + CITY_NAME + '/' + theDate2 + '.html'
        theurl2 = 'http://lishi.tianqi.com/' + CITY_NAME + '/' + str(vYear) + str(vMonth).zfill(2) + '.html'
        req = urllib2.Request(theurl2)
        response = urllib2.urlopen(req).read()
        response = response.decode('gbk').encode('utf-8')
        soup = BeautifulSoup(response, 'html.parser')
        reg = '<li>(.*)</li>'
        weatherList = re.compile(reg).findall(response)
        weather.append(weatherList)
        weather = weather[0]
        for k in xrange(100):
            j = 0
            for i in weather:
                if '<' in i:
                    del weather[j]
                if '（' in i:
                    del weather[j]
                j += 1
        weather = weather[6:]
        all_merge.extend(weather)
output = open('weather_list.txt', 'w')
for i in all_merge:
    output.write(i)
    output.write("\n")
output.close()

for i in xrange((end_time_timestamp - start_time_timestamp).days):
    final.ix[i, 1] = all_merge[5 * i]
    final.ix[i, 2] = all_merge[5 * i + 1]
    final.ix[i, 3] = all_merge[5 * i + 2]
    final.ix[i, 4] = all_merge[5 * i + 3]
    final.ix[i, 5] = all_merge[5 * i + 4]
final.to_csv(path_or_buf = 'weather_'+str(start_time)+'_'+str(end_time)+'.csv', index=None)
print ('Get all weather data completed, handling weather data')

#处理天气数据
data = pd.read_csv(filepath_or_buffer = 'weather_'+str(start_time)+'_'+str(end_time)+'.csv', header=0)
data['day'] = pd.to_datetime(data['day'])
data = data.fillna(str('-1'))
final = pd.DataFrame(np.zeros((len(data), 18), dtype=float), columns=['record_date', 'temp_max', 'temp_min', 'sunny', 'wind', 'rainy', 'cloudy', 'snowy', 'thunder', 'south', 'east', 'north', 'west', 'unknow', 'low', 'medium', 'high', 'super'])
final['record_date'] = data['day']
final['temp_max'] = data['tem_max']
final['temp_min'] = data['tem_min']
for i in xrange(len(data)):
    if str('晴') in data.ix[i, 3]:
        final.ix[i, 3] = 1
    if str('云') in data.ix[i, 3]:
        final.ix[i, 4] = 1
    if str('雨') in data.ix[i, 3]:
        final.ix[i, 5] = 1
    if str('阴') in data.ix[i, 3]:
        final.ix[i, 6] = 1
    if str('雪') in data.ix[i, 3]:
        final.ix[i, 7] = 1
    if str('雷') in data.ix[i, 3]:
        final.ix[i, 8] = 1
    if str('南') in data.ix[i, 4]:
        final.ix[i, 9] = 1
    if str('东') in data.ix[i, 4]:
        final.ix[i, 10] = 1
    if str('北') in data.ix[i, 4]:
        final.ix[i, 11] = 1
    if str('西') in data.ix[i, 4]:
        final.ix[i, 12] = 1
    if str('-1') in data.ix[i, 4]:
        final.ix[i, 13] = 1
    if str('小于3级') in data.ix[i, 5] or str('1级') in data.ix[i, 5] or str('2级') in data.ix[i, 5] or str('微风') in data.ix[i, 5]:
        final.ix[i, 14] = 1
    if str('3-4级') in data.ix[i, 5] or str('3级') == data.ix[i, 5]:
        final.ix[i, 15] = 1
    if str('4-5级') in data.ix[i, 5] or str('4级') == data.ix[i, 5]:
        final.ix[i, 16] = 1
    if str('5-6级') in data.ix[i, 5]:
        final.ix[i, 17] = 1
final.to_csv(path_or_buf = 'weather_handled.csv', index=None)
print ('Weather data is processed')
