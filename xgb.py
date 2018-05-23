# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:27:55 2017

@author: STP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb


def xgb_train(each_data, lr_predict_data):
    x = xgb.DMatrix(each_data.iloc[:, 1:], label=each_data.iloc[:, 0])
    predict = xgb.DMatrix(lr_predict_data)
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        #'gamma': 6,
        'slient': 1,
        'max_depth': 15,
        'eta': 0.05,
        'nthread': -1
    }
    xgb_model = xgb.train(params, x, num_boost_round=5000)
    xgb_output = xgb_model.predict(predict)
    return xgb_output


data = pd.read_csv('./train.csv', encoding='gbk')
sample = pd.read_csv('./example.csv')
weather_data = pd.read_csv('./weather_handled.csv')
del weather_data['unknow']
sample_shedid = sample.SHEDID.drop_duplicates()
shedid = data.SHEDID.drop_duplicates()
rtshedid = data.RTSHEDID.drop_duplicates()
result = pd.DataFrame(np.zeros((0, 0), dtype=int))
result2 = pd.DataFrame(np.zeros((0, 0), dtype=int))
data_merge = pd.DataFrame(np.zeros((0, 0), dtype=float))

weather_data.record_date = pd.to_datetime(weather_data.record_date)
weather_data['month'] = weather_data.record_date.apply(lambda x: x.month)
merge_data = weather_data.groupby('month').mean()[['temp_max', 'temp_min']].reset_index()
merge_data.columns = ['month', 'temp_max_avg', 'temp_min_avg']
weather_data = pd.merge(weather_data, merge_data, on='month', how='left')
weather_data.columns = ['leasetime', 'temp_max', 'temp_min', 'sunny', 'wind', 'rainy', 'cloudy', 'snowy', 'thunder', 'south', 'east', 'north', 'west', 'low', 'medium', 'high', 'super', 'month', 'temp_max_avg', 'temp_min_avg']
weather_data['weekend'] = weather_data.leasetime.apply(lambda x: x.dayofweek)
weekend_dummy = pd.get_dummies(weather_data['weekend'], prefix='weekend')
weather_data = pd.concat([weather_data, weekend_dummy], axis=1)
del weather_data['weekend']
weather_data['temp-avg-max'] = weather_data.temp_max - weather_data.temp_max_avg
weather_data['temp-avg-min'] = weather_data.temp_min - weather_data.temp_min_avg
temp_max_shift = pd.DataFrame(weather_data.temp_max.shift(1).fillna(6))
temp_min_shift = pd.DataFrame(weather_data.temp_min.shift(1).fillna(-4))
temp_max_shift.columns = ['temp_max_mins']
temp_min_shift.columns = ['temp_min_mins']
weather_data = pd.concat([weather_data, temp_max_shift, temp_min_shift], axis=1)
weather_data.temp_max_mins = weather_data.temp_max - weather_data.temp_max_mins
weather_data.temp_min_mins = weather_data.temp_min - weather_data.temp_min_mins
weather_data['mins_sum'] = weather_data.temp_max_mins + weather_data.temp_min_mins
weather_data['temp-avg-sum'] = weather_data['temp-avg-max'] + weather_data['temp-avg-min']
columns_list = []
for i in ['sunny', 'wind', 'rainy', 'cloudy']:
    for j in weekend_dummy.columns:
        weather_data[str(i) + '_' + str(j)] = weather_data[i] * weather_data[j]
        if (i != str('cloudy')):
            if (j == str('weekend_5')) or (j == str('weekend_6')):
                columns_list.append([str(i) + '_' + str(j)])
for i in ['temp-avg-max', 'temp-avg-min', 'temp-avg-sum', 'temp_max_mins', 'temp_min_mins', 'mins_sum']:
    for j in columns_list:
        weather_data[str(i) + '_' + str(j[0])] = weather_data[i] * weather_data[j[0]]

data.LEASEDATE = pd.to_datetime(data.LEASEDATE)
data['leasetime'] = data.LEASEDATE + pd.to_timedelta(data.LEASETIME)
data_show = []
weekend7 = pd.date_range(start='2015-09-05', end='2015-10-31', freq='7D')
weekend6 = pd.date_range(start='2015-09-06', end='2015-10-31', freq='7D')
low = 0
up = 0
avg_number = 0
median_number = 0
# plt.figure()
lr_predict_data = weather_data[len(weather_data) - 61:]
del lr_predict_data['leasetime']
for j, i in enumerate(sample_shedid.values, start=1):
    if i in shedid.values:
        each_data = data[data.SHEDID == i]
        each_data['show'] = 1
        each_data = each_data[['leasetime', 'show']].set_index('leasetime')
        each_data = each_data.groupby(pd.Grouper(freq='1D')).sum().fillna(0)
        data_show.append([i, len(each_data)])
        if len(each_data) < (2 * 7):
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['LEASE'] = each_data.values.mean() * 1.25
            data_merge.LEASE = data_merge[['time', 'LEASE']].apply(lambda x: 1 * x.LEASE if x.time in weekend6 or x.time in weekend7 else x.LEASE, axis=1)
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
            result = pd.concat([result, data_merge], axis=0)
            low += 1
            continue

        if len(each_data) > (9 * 7):
#            drop_time = pd.date_range(start='20150630', end='20150831')
#            for drop in drop_time:
#                try:
#                    each_data = each_data.drop(drop)
#                except:
#                    pass
            drop_time2 = pd.date_range(start='20150101', end='20150228')
            for drop in drop_time2:
                try:
                    each_data = each_data.drop(drop)
                except:
                    pass
            each_data = each_data.reset_index()
            each_data = each_data.merge(weather_data, on='leasetime', how='left')
            del each_data['leasetime']
            xgb_predict = xgb_train(each_data, lr_predict_data)
#            for day in range(7):
#                xgb_predict[30+day] *= 0.75
#            plt.figure()
#            plt.plot(xgb_predict)
#            plt.figure()
#            plt.plot(pd.date_range(end='20150831',periods=len(each_data)), each_data.show.values)
#            plt.plot(pd.date_range(start='20150901',periods=61), xgb_predict) #orange
#            plt.show()
                
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['LEASE'] = xgb_predict
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
            result = pd.concat([result, data_merge], axis=0)
            median_number += 1
            print('handled lease number: %s/%s, xgb' % (j, len(sample_shedid)))
            continue

        repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
        repeat = np.array(repeat[:61]) * 1.25

        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['SHEDID'] = i
        data_merge['LEASE'] = repeat
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
        result = pd.concat([result, data_merge], axis=0)
        avg_number += 1
    else:
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['LEASE'] = 0
        data_merge.LEASE = data_merge[['time', 'LEASE']].apply(lambda x: 1 * x.LEASE if x.time in weekend6 or x.time in weekend7 else x.LEASE, axis=1)
        data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
        result = pd.concat([result, data_merge], axis=0)
    print('handled lease number: %s/%s, less' % (j, len(sample_shedid)))
low2 = 0
up2 = 0
avg_number2 = 0
median_number2 = 0
for j, i in enumerate(sample_shedid.values, start=1):
    if i in shedid.values:
        each_data = data[data.RTSHEDID == i]
        each_data['show'] = 1
        each_data = each_data[['leasetime', 'show']].set_index('leasetime')
        each_data = each_data.groupby(pd.Grouper(freq='1D')).sum().fillna(0)

        if len(each_data) < (2 * 7):
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['RT'] = each_data.values.mean() * 1.25
            data_merge.RT = data_merge[['time', 'RT']].apply(lambda x: 1 * x.RT if x.time in weekend6 or x.time in weekend7 else x.RT, axis=1)
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'RT']]
            result2 = pd.concat([result2, data_merge], axis=0)
            low2 += 1
            continue

        if len(each_data) > (9 * 7):
#            drop_time = pd.date_range(start='20150630', end='20150831')
#            for drop in drop_time:
#                try:
#                    each_data = each_data.drop(drop)
#                except:
#                    pass
            drop_time2 = pd.date_range(start='20150101', end='20150228')
            for drop in drop_time2:
                try:
                    each_data = each_data.drop(drop)
                except:
                    pass
            each_data = each_data.reset_index()
            each_data = each_data.merge(weather_data, on='leasetime', how='left')
            del each_data['leasetime']
            xgb_predict = xgb_train(each_data, lr_predict_data)

            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['RT'] = xgb_predict
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'RT']]
            result2 = pd.concat([result2, data_merge], axis=0)
            median_number2 += 1
            print('handled rt number: %s/%s, xgb' % (j, len(sample_shedid)))
            continue

        repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
        repeat = np.array(repeat[:61]) * 1.25

        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['SHEDID'] = i
        data_merge['RT'] = repeat
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge = data_merge[['SHEDID', 'time', 'RT']]
        result2 = pd.concat([result2, data_merge], axis=0)
        avg_number2 += 1
    else:
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['RT'] = 0
        data_merge.RT = data_merge[['time', 'RT']].apply(lambda x: 1 * x.RT if x.time in weekend6 or x.time in weekend7 else x.RT, axis=1)
        data_merge = data_merge[['SHEDID', 'time', 'RT']]
        result2 = pd.concat([result2, data_merge], axis=0)
    print('handled rt number: %s/%s, less' % (j, len(sample_shedid)))
print('lease: low:%s, avg:%s, median:%s, up:%s' % (low, avg_number, median_number, up))
print('rt: low:%s, avg:%s, median:%s, up:%s' % (low2, avg_number2, median_number2, up2))
sub = pd.merge(result, result2, on=['SHEDID', 'time'], how='outer')
sub = sub[['SHEDID', 'time', 'RT', 'LEASE']]
sub = pd.merge(sample[['SHEDID', 'time']], sub, on=['SHEDID', 'time'], how='left')
#sub.RT = sub.RT.apply(lambda x: 12 if x < 12 else x)
#sub.LEASE = sub.LEASE.apply(lambda x: 12 if x < 12 else x)

sub.time = pd.to_datetime(sub.time)
festival = pd.date_range(start='20151001', end='20151007')
merge_RT = sub[['time', 'RT']].apply(lambda x: 0.75 * x.RT if x.time in festival else x.RT, axis=1)
merge_LEASE = sub[['time', 'LEASE']].apply(lambda x: 0.75 * x.LEASE if x.time in festival else x.LEASE, axis=1)
sub = pd.concat([sub[['SHEDID', 'time']], merge_RT, merge_LEASE], axis=1)

sub.columns = ['SHEDID', 'time', 'RT', 'LEASE']
sub.time = sub.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
#sub.to_csv('predict2.csv', index=None)
