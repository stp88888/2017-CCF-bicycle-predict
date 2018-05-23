# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:45:08 2017

@author: STP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./train.csv', encoding='gbk')
sample = pd.read_csv('./example.csv')
avg_week = 3
sample_shedid = sample.SHEDID.drop_duplicates()
shedid = data.SHEDID.drop_duplicates()
rtshedid = data.RTSHEDID.drop_duplicates()
result = pd.DataFrame(np.zeros((0, 0), dtype=int))
result2 = pd.DataFrame(np.zeros((0, 0), dtype=int))
data_merge = pd.DataFrame(np.zeros((0, 0), dtype=float))
data['leasetime'] = pd.to_datetime(data.LEASEDATE) + pd.to_timedelta(data.LEASETIME)
weekend7 = pd.date_range(start='2015-09-05', end='2015-10-31', freq='7D')
weekend6 = pd.date_range(start='2015-09-06', end='2015-10-31', freq='7D')
weekend_rain = ['20150905','20150906']
weekend_cold = ['20150912','20150913','20151010','20151011','20151031']
weekend_rain = pd.to_datetime(weekend_rain)
weekend_cold = pd.to_datetime(weekend_cold)
cold_parameter = 1
rain_parameter = 1
data_show = []
low = 0
middle_num = 0
avg_number = 0
median_number = 0
#plt.figure()
for j, i in enumerate(sample_shedid.values, start=1):
    if i in shedid.values:
        each_data = data[data.SHEDID == i]
        each_data['show'] = 1
        each_data = each_data[['leasetime', 'show']].set_index('leasetime')
        each_data = each_data.groupby(pd.Grouper(freq='1D')).sum().fillna(0)
        #plt.plot(each_data)
        data_show.append([i, len(each_data), 'lease'])
        
        if len(each_data) < (2 * 7):
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['LEASE'] = each_data.values.mean() * 1.3
            data_merge.LEASE = data_merge[['time', 'LEASE']].apply(lambda x:1 * x.LEASE if x.time in weekend6 or x.time in weekend7 else x.LEASE, axis=1)
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
            result = pd.concat([result, data_merge], axis=0)
            low += 1
            continue
        
        if len(each_data) > (avg_week * 8 * 7):
            yushu = len(each_data) % 7
            each_data_7days = each_data.iloc[yushu:]
            each_data_7days = pd.DataFrame(each_data_7days.values.reshape(int(len(each_data_7days)/7),7))
            start_index = each_data_7days.shape[0] - (avg_week * 8)
            each_data_handle = each_data_7days.iloc[start_index:(start_index+5*avg_week),:]
            each_data_7days_median = each_data_handle.mean()  # without weather data, remove the effects of the weather
            repeat = np.array(61 * list(each_data_7days_median.values))[:61]
            
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['LEASE'] = repeat
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
            result = pd.concat([result, data_merge], axis=0)
            median_number += 1
            continue

#        if len(each_data) > (3 * 7):
#            yushu = len(each_data) % 7
#            each_data_7days = each_data.iloc[yushu:]
#            each_data_7days = pd.DataFrame(each_data_7days.values.reshape(int(len(each_data_7days)/7),7))
#            each_data_7days_median = each_data_7days.iloc[-3:,:].mean()  # without weather data, remove the effects of the weather
#            repeat1 = np.array(61 * list(each_data_7days_median.values))[:61]
#            repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
#            repeat2 = np.array(repeat[:61]) * 1.25
#            repeat = 0.5 * repeat1 + 0.5 * repeat2
#            
#            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
#            data_merge['SHEDID'] = i
#            data_merge['LEASE'] = repeat
#            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
#            data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
#            result = pd.concat([result, data_merge], axis=0)
#            middle_num += 1
#            continue

        repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
        repeat = np.array(repeat[:61]) * 1.3
        
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['LEASE'] = repeat
        data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
        result = pd.concat([result, data_merge], axis=0)
        avg_number += 1
    else:
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['LEASE'] = 0
        data_merge = data_merge[['SHEDID', 'time', 'LEASE']]
        result = pd.concat([result, data_merge], axis=0)
    print('handled lease number: %s/%s' % (j, len(sample_shedid)))
low2 = 0
middle_num2 = 0
avg_number2 = 0
median_number2 = 0
for j, i in enumerate(sample_shedid.values, start=1):
    if i in shedid.values:
        each_data = data[data.RTSHEDID == i]
        each_data['show'] = 1
        each_data = each_data[['leasetime', 'show']].set_index('leasetime')
        each_data = each_data.groupby(pd.Grouper(freq='1D')).sum().fillna(0)
        data_show.append([i, len(each_data), 'rt'])
        
        if len(each_data) < (2 * 7):
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['SHEDID'] = i
            data_merge['RT'] = each_data.values.mean() * 1.3
            data_merge.RT = data_merge[['time', 'RT']].apply(lambda x:1 * x.RT if x.time in weekend6 or x.time in weekend7 else x.RT, axis=1)
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge = data_merge[['SHEDID', 'time', 'RT']]
            result2 = pd.concat([result2, data_merge], axis=0)
            low2 += 1
            continue
        
        if len(each_data) > (avg_week * 8 * 7):
            yushu = len(each_data) % 7
            each_data_7days = each_data.iloc[yushu:]
            each_data_7days = pd.DataFrame(each_data_7days.values.reshape(int(len(each_data_7days)/7),7))
            start_index = each_data_7days.shape[0] - (avg_week * 8)
            each_data_handle = each_data_7days.iloc[start_index:(start_index+avg_week * 5),:]
            each_data_7days_median = each_data_handle.mean()  # without weather data, remove the effects of the weather
            repeat = np.array(61 * list(each_data_7days_median.values))[:61]
            
            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
            data_merge['SHEDID'] = i
            data_merge['RT'] = repeat
            data_merge = data_merge[['SHEDID', 'time', 'RT']]
            result2 = pd.concat([result2, data_merge], axis=0)
            median_number2 += 1
            continue
            
#        if len(each_data) > (3 * 7):
#            yushu = len(each_data) % 7
#            each_data_7days = each_data.iloc[yushu:]
#            each_data_7days = pd.DataFrame(each_data_7days.values.reshape(int(len(each_data_7days)/7),7))
#            each_data_7days_median = each_data_7days.iloc[-3:,:].mean()  # without weather data, remove the effects of the weather
#            repeat1 = np.array(61 * list(each_data_7days_median.values))[:61]
#            repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
#            repeat2 = np.array(repeat[:61]) * 1.25
#            repeat = 0.5 * repeat1 + 0.5 * repeat2
#            
#            data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
#            data_merge['SHEDID'] = i
#            data_merge['RT'] = repeat
#            data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
#            data_merge = data_merge[['SHEDID', 'time', 'RT']]
#            result2 = pd.concat([result2, data_merge], axis=0)
#            middle_num2 += 1
#            continue
        
        repeat = 10 * list(each_data.values[len(each_data) - 2 * 7:].reshape(2, 7).mean(axis=0))
        repeat = np.array(repeat[:61]) * 1.3
        
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['RT'] = repeat
        data_merge = data_merge[['SHEDID', 'time', 'RT']]
        result2 = pd.concat([result2, data_merge], axis=0)
        avg_number2 += 1
    else:
        data_merge['time'] = pd.date_range(start='2015-09-01', periods=61, freq='D')
        data_merge['time'] = data_merge.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
        data_merge['SHEDID'] = i
        data_merge['RT'] = 0
        data_merge = data_merge[['SHEDID', 'time', 'RT']]
        result2 = pd.concat([result2, data_merge], axis=0)
    print('handled rt number: %s/%s' % (j, len(sample_shedid)))
print ('lease: low:%s, avg:%s, median:%s, middle_num:%s' % (low, avg_number, median_number, middle_num))
print ('rt: low:%s, avg:%s, median:%s, middle_num2:%s' % (low2, avg_number2, median_number2, middle_num2))
data_show = pd.DataFrame(data_show)

sub = pd.merge(result, result2, on=['SHEDID', 'time'], how='outer')
sub = sub[['SHEDID', 'time', 'RT', 'LEASE']]
sub = pd.merge(sample[['SHEDID', 'time']], sub, on=['SHEDID', 'time'], how='left')
sub.RT = sub.RT.apply(lambda x: 12 if x < 12 else x)
sub.LEASE = sub.LEASE.apply(lambda x: 12 if x < 12 else x)

sub.time = pd.to_datetime(sub.time)
festival = pd.date_range(start='20151001', end='20151007')
merge_RT = sub[['time', 'RT']].apply(lambda x: 0.75 * x.RT if x.time in festival else x.RT, axis=1)
merge_LEASE = sub[['time', 'LEASE']].apply(lambda x: 0.75 * x.LEASE if x.time in festival else x.LEASE, axis=1)
sub = pd.concat([sub[['SHEDID', 'time']], merge_RT, merge_LEASE], axis=1)
sub.columns = ['SHEDID', 'time', 'RT', 'LEASE']
sub.LEASE = sub[['time', 'LEASE']].apply(lambda x:x.LEASE * cold_parameter if x.time in weekend_cold else x.LEASE, axis=1)
sub.RT = sub[['time', 'RT']].apply(lambda x:x.RT * cold_parameter if x.time in weekend_cold else x.RT, axis=1)
sub.LEASE = sub[['time', 'LEASE']].apply(lambda x:x.LEASE * rain_parameter if x.time in weekend_rain else x.LEASE, axis=1)
sub.RT = sub[['time', 'RT']].apply(lambda x:x.RT * rain_parameter if x.time in weekend_rain else x.RT, axis=1)
down=pd.date_range(start='20151031', periods=1)
merge_RT = sub[['time', 'RT']].apply(lambda x: 0.75 * x.RT if x.time in down else x.RT, axis=1)
merge_LEASE = sub[['time', 'LEASE']].apply(lambda x: 0.75 * x.LEASE if x.time in down else x.LEASE, axis=1)
sub.time = sub.apply(lambda x: str(x['time'].year) + str('/') + str(x['time'].month) + str('/') + str(x['time'].day), axis=1)
#sub.to_csv('predict.csv', index=None)