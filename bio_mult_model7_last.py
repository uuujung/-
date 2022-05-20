import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import time
import pandas as pd
import pickle
import tensorflow as tf
import kde as KDE
import random

#import db_io_utils as utils_
tf.random.set_seed(3172021)


gpu = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)

#import db_io_utils as utils_

import Model7 as Model


'''
ys_intp_data, ys_nw_data, ys_raw_data, variables, raw_field_dict = pickle.load(open('ys_data_.tlp', 'rb'))

df = ys_intp_data.iloc[329:]

n_in = list()
for i in range(len(df)):
    n_in.append(i)

df['index'] = n_in
df.index = df['index']
'''

'''
ys_intp_data, ys_nw_data, ys_raw_data, variables, raw_field_dict = pickle.load(open('ys_data_.tlp', 'rb'))
df = ys_nw_data
df2 = ys_intp_data
features = ['OLR(kg VS/m3)', 'kg_VSadd', 'VFA', 'Alkalinity', 'VFA/Alk ratio', 'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']
raw_data = df[features]
raw_data2 = df2[features]

learning_size = 240 #240
vsadd_mean = list()
for i in range(len(raw_data) - learning_size +1 ):
    data = raw_data.iloc[ 0 + i : learning_size + i ]
    Vsadd_mean = np.mean(data.kg_VSadd)
    vsadd_mean.append(Vsadd_mean)
    
Vsadd_mean_learning = pd.DataFrame(vsadd_mean, columns=['vsadd_mean'])

#Vsadd_mean은 앞 러닝사이즈개 만큼의 평균을 구한거라서 처음 러닝사이즈개 만큼은 데이터가 없음. 그 인덱스 맞추려고 한것.
n_in = list()
for i in range(learning_size-1 , len(raw_data)):
    n_in.append(i)

Vsadd_mean_learning.index = n_in

OLR = raw_data.iloc[learning_size-1 : ]['OLR(kg VS/m3)']
#OLR = pd.DataFrame(OLR)

data = pd.concat([Vsadd_mean_learning, OLR], axis = 1)
new_Vsadd = data['OLR(kg VS/m3)'] * data['vsadd_mean']
new_Vsadd = pd.DataFrame(new_Vsadd , columns = ['new_Vsadd'])


Production = raw_data.iloc[learning_size-1 : ]['Biogas Production(m3 x 4)']
#a.transpose

#Production = pd.DataFrame(Production)
for_mean = pd.concat([new_Vsadd, Production], axis=1)

new_yield = for_mean['Biogas Production(m3 x 4)'] / for_mean['new_Vsadd']
new_yield = pd.DataFrame(new_yield, columns = ['new_yield'])

new_raw_data = pd.concat([df['Time'], raw_data, Vsadd_mean_learning, new_yield, new_Vsadd], axis = 1) #나다라야
new_raw_data2 = pd.concat([df2['Time'], raw_data2, Vsadd_mean_learning, new_yield, new_Vsadd], axis = 1) #리니어
#new_raw_data.index = df['Time']

df = new_raw_data.iloc[329:] #NW
df2 = new_raw_data2.iloc[329:] #리니어

n_in = list()
for i in range(len(df)):
    n_in.append(i)

df.index = n_in
df2.index = n_in

'''




#####DB로 데이터 불러오기
'''
param = utils_.get_config() #db_io_utils 의 get_config 함수 사용해서 config.ini 읽어오기(conrig 파일을 사이트별로 맞게 수정 하면 됨)
conn_inf = [param['host'], param['user'], param['password'], param['database'], param['port']]

site = param['site'] #config 파일의 site를 site로 지정

raw_data_table = param['raw_data_table'] #config 파일의 raw_data_table을 raw_data_table로 지정
data_table = param['data_table']
pred_table = param['pred_table']

db_field_list = param['db_field_list']
db_field_type = param['db_field_type']

raw_field_dict = param['raw_field'] # config 파일의 raw_field를 raw_field_dict으로 지정
pred_features_dict_ = param['pred_features']
pred_field_dict = utils_.make_pred_field_info(raw_field_dict, param['pred_field'], pred = True)  # time, olr 제외
pred_features_dict = utils_.make_pred_field_info(raw_field_dict, param['pred_features'], pred=False)  # 실제 예측할 변수

intp_field_dict = utils_.make_field_info_suffix(raw_field_dict, '_intp')
nw_field_dict = utils_.make_field_info_suffix(raw_field_dict, '_nw')
nw_intp_field_dict = utils_.make_field_info_suffix(raw_field_dict, '_nw_intp')

# update_field_list = intp_field_list + nw_field_list + nw_intp_field_list
update_field_dict = dict()
update_field_dict.update(intp_field_dict)
update_field_dict.update(nw_field_dict)
update_field_dict.update(nw_intp_field_dict)

# data_field_list = raw_field_list + update_field_list
data_field_dict = dict()
data_field_dict.update(raw_field_dict)
data_field_dict.update(update_field_dict)




ys_intp_data = utils_.load_data(conn_inf, site, data_table, intp_field_dict)
ys_nw_data = utils_.load_data(conn_inf, site, data_table, nw_intp_field_dict)
ys_raw_data = utils_.load_data(conn_inf, site, raw_data_table, raw_field_dict)

#ys_intp_data.columns = ['Time','vsadd','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield']
#df = ys_intp_data
ys_nw_data.columns = ['Time','vsadd','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield']
df = ys_nw_data




yusan_intp_data = utils_.load_data(conn_inf, site, data_table, intp_field_dict)
yusan_nw_data = utils_.load_data(conn_inf, site, data_table, nw_intp_field_dict)
yusan_raw_data = utils_.load_data(conn_inf, site, raw_data_table, raw_field_dict)
#variables = param['vars']

#yusan_intp_data.columns = ['Time','vsadd','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield']
#df = yusan_intp_data
yusan_nw_data.columns = ['Time','vsadd','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield']
df = yusan_nw_data






metro_intp_data = utils_.load_data(conn_inf, site, data_table, intp_field_dict)
metro_nw_data = utils_.load_data(conn_inf, site, data_table, nw_intp_field_dict)
metro_raw_data = utils_.load_data(conn_inf, site, raw_data_table, raw_field_dict)

metro_intp_data.columns =['Time','olr','nh4_n', 'vfa', 'alk', 'ph', 'biogas_production', 'biogas_yield']
df = metro_intp_data
#metro_nw_data.columns = ['Time','vsadd','olr', 'vfa', 'alk', 'vfa_alk', 'biogas_production', 'biogas_yield']
#df = metro_nw_data

'''

metro_intp_data, metro_nw_data, metro_raw_data, metro_field_dict = pickle.load(open('metro.tlp', 'rb'))

df2 = metro_nw_data.iloc[19:]
df = metro_intp_data[79:]


class Scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, embedding_dim, start_step, warmup_steps = 2500):
      
        super(Scheduler, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding_dim = tf.cast(self.embedding_dim, tf.float32)
        
        self.start_step = start_step
        self.warmup_steps = warmup_steps
    
    
    def __call__(self, step):

        step += self.start_step
        
        value1 = tf.math.rsqrt(step)
        value2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(value1, value2) + 1E-5





df.columns


restore = './model2/test_model'

learning_size = 180 #240
window_size = 30

d_model = 6
in_dim = 1 +1
key_dim = 4
out_dim = 1

gap = 5
step = 1

data_size = learning_size + gap


num_layers = 1

batch_size = 40 #60
#learning_rate = 1E-2
patience = 10

#ys_pickle_vsadd
features = ['OLR(kg VS/m3)', 'NH4_N', 'VFA', 'Alkalinity', 'PH', 'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']

#features = pred_features_dict_[1:]

raw_data = df[features]
raw_data2 = df2[features]


input_index = [0]
key_index = [1, 2, 3, 4]
target_index = [5]
    
raw_data = df[features]
raw_data.index = df['Time']

x_data = raw_data.iloc[:-gap, input_index]
x_data['Time'] = list(np.arange(1, df.shape[0] - gap + 1 ))
x_data.index = df['Time'][:-gap]

kt_data = raw_data.iloc[:-gap, key_index]
kt_data.index = df['Time'][:-gap]

k_data = raw_data.iloc[gap  :, key_index]
k_data.index = df['Time'][gap :]

y_data = raw_data.iloc[gap :, target_index]
y_data.index = df['Time'][gap:]

'''
y = y_data.iloc[:, 1]

plt.plot(y)
plt.show()
'''




warm_up = 0
test_period = 30

be_loss = list()
true = list()
pred = list()


true_ = list()
pred_ = list()
mape_ = list()



#for date in range(60, 180 + test_period + warm_up, 1):
    
for date in range(0, test_period, 1): #test_period, 1):    
#date = 0
    start_time = time.time()    


    #if date_ < 0:
    #    date = 0
    #else:
    #    date = date_  + 385

    #date = 0 
    data_position = data_size + date 
    #print(data_position, '\n')#, y_data.iloc[data_position, :])

    x_data_for_i = x_data.iloc[date:(data_position), :] #9월2일부터 시작
    x_data_for_i.index = x_data.index[date:(data_position)]

    #weights = np.sqrt(range(1, learning_size + 1))
    #x_data_for_i['Weights'] = [w for w in weights] + [weights[-1]] * gap

    #plt.plot(weights)
    #plt.show()

    kt_data_for_i = kt_data.iloc[date:(data_position), :]
    kt_data_for_i.index = kt_data.index[date:(data_position)]

    k_data_for_i = k_data.iloc[date:(data_position), :]
    k_data_for_i.index = k_data.index[date:(data_position)]

    y_data_for_i = y_data.iloc[date:(data_position), :]
    y_data_for_i.index = y_data.index[date:(data_position)]
    
    
    #print(y_data_for_i.index[-1])
    
    
    # print('<Yesterday>\n\n', y_data_for_i.iloc[239, :], '\n')
    # print('<Today>\n\n', y_data_for_i.iloc[240, :], '\n')
    # print('<Point to be predicted at today>\n\n', y_data_for_i.iloc[245, :], '\n')
    
    #print(x_data_for_i.index[-1])
    
    # print('<Training date for yesterday>\n\n', x_data_for_i.iloc[239, :], '\n')
    # print('<Today for prediction X>\n\n', x_data_for_i.iloc[240, :], '\n')
    # print('<Today for prediction Y>\n\n', y_data_for_i.iloc[240, :], '\n')


    # train data    
    x_train_for_i = x_data_for_i.iloc[:-gap]
    kt_train_for_i = kt_data_for_i.iloc[:-gap]
    k_train_for_i = k_data_for_i.iloc[:-gap]
    y_train_for_i = y_data_for_i.iloc[:-gap]
    
    #print(y_train_for_i.iloc[-1, :])
    #print(x_train_for_i.iloc[-1, :])

    # train data standardize
    x_train_mean_for_i = x_train_for_i.mean()
    x_train_std_for_i = x_train_for_i.std()

    k_train_mean_for_i = k_train_for_i.mean()
    k_train_std_for_i = k_train_for_i.std()

    kt_train_mean_for_i = kt_train_for_i.mean()
    kt_train_std_for_i = kt_train_for_i.std()

    y_train_mean_for_i = y_train_for_i.mean()
    y_train_std_for_i = y_train_for_i.std()


    x_train_for_i_ = (x_train_for_i - x_train_mean_for_i) / (x_train_std_for_i + 1E-8)
    kt_train_for_i_ = (kt_train_for_i - kt_train_mean_for_i) / (kt_train_std_for_i + 1E-8)
    k_train_for_i_ = (k_train_for_i - k_train_mean_for_i) / (k_train_std_for_i + 1E-8)
    y_train_for_i_ = (y_train_for_i - y_train_mean_for_i) / (y_train_std_for_i + 1E-8)


    # data for mape
    true_k = k_data_for_i.iloc[-1, :]
    true_y = y_data_for_i.iloc[-1, :]
    
    # cast for tensor
    x_train_for_i_ = x_train_for_i_.values.astype(np.float32)
    kt_train_for_i_ = kt_train_for_i_.values.astype(np.float32)
    k_train_for_i_ = k_train_for_i_.values.astype(np.float32)
    y_train_for_i_ = y_train_for_i_.values.astype(np.float32)

    true_k_ = true_k.values.astype(np.float32)
    true_y_ = true_y.values.astype(np.float32)


    # windowing
    x_train = list()
    k_train = list()
    kt_train = list()
    y_train = list()
    
    (len(x_train_for_i) - window_size) / step + 1
    
    for pos in range(int((len(y_train_for_i) - window_size) / step + 1)):

        x_train.append(x_train_for_i_[pos:(window_size + pos)])
        kt_train.append(kt_train_for_i_[pos:(window_size + pos)])
        k_train.append(k_train_for_i_[pos:(window_size + pos)])
        y_train.append(y_train_for_i_[pos:(window_size + pos)])

    
    
    # tensor pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, kt_train, k_train, y_train))
    train_dataset = train_dataset.cache().shuffle(len(x_train)).repeat().batch(batch_size)

    # Model init states
    tinit_states = tf.convert_to_tensor(np.zeros([2, batch_size, d_model], dtype = 'float32'), dtype = tf.float32)
    pinit_states = tf.convert_to_tensor(np.zeros([2, 1, d_model], dtype = 'float32'), dtype = tf.float32)

    # iepoch number
    train_take = len(x_train) // batch_size

    #inputs, tkeys, keys, outputs = next(iter(train_dataset))


    model = Model.construct(in_dim, key_dim, out_dim, d_model, num_layers, False)

    learning_rate = 0.01#Scheduler(d_model, 0, train_take)

    # plt.plot(learning_rate(tf.range(train_take * 50, dtype = tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()

    optimizer = tf.keras.optimizers.Adam(learning_rate)


    if date > 0:
        print('\n\tLoad weights...', end = '')
        model.load_weights(restore)
        print('Done\n')


    previous_loss = 0
    best_loss = 1E+8
    epoch_loss = 0.
    early_stop_step = 0
    epoch = 1
    

    for iepoch, train_data in enumerate(train_dataset.take(train_take * 500)): #
    
        inputs, tkeys, keys, outputs = train_data


        with tf.GradientTape() as tape:
                
            in_key_outputs, in_key_out_outputs = model(inputs, tkeys, keys, outputs, tinit_states)
            
            
            loss = 0.75 * tf.reduce_mean(tf.reduce_sum(tf.abs((in_key_outputs - keys) / keys) * tf.constant([1/4., 1/4., 1/4., 1/4.]), -1)) * 100. \
                                  + 0.25 * tf.reduce_mean(tf.reduce_sum(tf.abs((in_key_out_outputs - outputs) / outputs) , -1)) * 100.
            
        
            #loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.abs((in_key_outputs - keys) / keys) * tf.constant([1/4., 1/4., 1/4., 1/4.]), -1)) * 100. \
            #                      + 0.5 * tf.reduce_mean(tf.reduce_sum(tf.abs((in_key_out_outputs - outputs) / outputs) * tf.constant([0.5,0.5]), -1)) * 100.
                                  # 여기서 tf.reduce_sum -1 해주는건 행 합 하라는거임. 즉 행 갯수만큼의 결과값 나오게됨
                                  
#tf.reduce_mean(tf.reduce_sum(tf.abs((in_key_outputs - keys) / keys) * tf.constant([1. / 3, 1. / 3., 1. / 3.]), -1)) * 100.
#tf.reduce_mean(tf.abs((in_key_outputs - keys)/keys) * tf.constant([1. / 3, 1. / 3., 1. / 3.]))*100

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += loss

        print('iepoch %d: loss = %.4f' % (iepoch + 1, loss.numpy()))


        if (iepoch + 1) % train_take == 0:
            
            epoch_loss /= train_take
        

            if best_loss < epoch_loss:
                early_stop_step += 1

                if early_stop_step > patience:
                    print('\t\t===> Early stopped at epoch %d: loss %.4f(%.4f) took %.4f' % (epoch, epoch_loss, best_loss, time.time() - start_time))
                    model.load_weights(restore)
                    be_loss.append(best_loss)
                    #print(be_loss[-1])
                    break
            else:
                early_stop_step = 0
                best_loss = epoch_loss
                model.save_weights(restore)


            crit = abs(previous_loss - epoch_loss) 
            
            print('\t\t===> Epoch %d: loss %.4f(%.4f)' % (epoch, epoch_loss, best_loss))
            
            
            if crit < 1E-4 * previous_loss:
                
                print('\n\t\t===> Converged at epoch %d', epoch)
                be_loss.append(best_loss)
                break
            
            previous_loss = epoch_loss

            epoch_loss = 0.
            epoch += 1


    if date >= warm_up:
        #pred 원본
        #=====================================================================
        #k_pred_for_i_ = k_train_for_i_
        #k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...] # batch axis append

        #y_pred_for_i_ = y_train_for_i_
        #y_pred_for_i_ = y_pred_for_i_[np.newaxis, ...] # batch axis append
        
        #=====================================================================
         


        #pred NW ver
        #=====================================================================
        #k_pred_for_i = k_train_for_i.reset_index().rename(columns={"Time": "RegDate"})
         
        #kde = KDE.KDE(k_pred_for_i)
        #jackknife = kde.b_jackknife(start = 0.1, step = 10)
        #num_bins = kde.num_bins(jackknife, start = 1, stop = 50, step = 5)
         
        #NW_data = kde.NW_predict(num_bins, jackknife) #num_bins, b_jackknife
        #nw_data, nw_intp_data = kde.NW_interpolation(NW_data)
         
        #k_nw_intp_data_ = pd.DataFrame(nw_intp_data, columns = ['RegDate',  'NH4_N', 'VFA', 'Alkalinity', 'PH'])#, 'VFA/Alk ratio']) #원데이터 + 보간 값
        #k_nw_intp_data_ = k_nw_intp_data_.drop('RegDate', axis=1)
        #k_nw_intp_data_.index = k_data_for_i.reset_index()['Time']
        #k_nw_intp_data_= k_nw_intp_data_ .values.astype(np.float32) 
        #k_pred_for_i_ = k_nw_intp_data_[np.newaxis, ...]

         
        #y_pred_for_i = y_train_for_i.reset_index().rename(columns={"Time": "RegDate"})
        
        #kde_y = KDE.KDE(y_pred_for_i)
        #jackknife_y = kde_y.b_jackknife(start = 0.1, step = 10)
        #num_bins_y = kde_y.num_bins(jackknife_y, start = 1, stop = 50, step = 5)
         
        #NW_data_y = kde_y.NW_predict(num_bins_y, jackknife_y) #num_bins,b_jackknife
        #nw_data_y, nw_intp_data_y = kde_y.NW_interpolation(NW_data_y)
         
        #y_nw_intp_data_ = pd.DataFrame(nw_intp_data_y, columns = ['RegDate', 'Biogas Production(m3 x 4)'])#, 'new_yield'])#원데이터 + 보간 값
        #y_nw_intp_data_ = y_nw_intp_data_.drop('RegDate', axis=1)
        #y_nw_intp_data_.index = y_data_for_i.reset_index()['Time']
        #y_nw_intp_data_= y_nw_intp_data_ .values.astype(np.float32) 
        #y_pred_for_i_ = y_nw_intp_data_[np.newaxis, ...]     
           
        #=====================================================================
        

        #pred 마지막 5일 반복 ver
        #=====================================================================

        #k_pred_for_i_ = pd.concat((k_train_for_i,  k_train_for_i[-5:])).values
        #k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...]

        #y_pred_for_i_ = pd.concat((y_train_for_i,  y_train_for_i[-5:])).values
        #y_pred_for_i_ = y_pred_for_i_[np.newaxis, ...]     
        #=====================================================================


        #pred 마지막 5일 평균
        #=====================================================================

        k_pred_for_i_ = k_train_for_i
        
        for i in range(5):
            data = k_pred_for_i_.iloc[-5  : ]
            k_mean = np.mean(data)
            k_mean = k_mean[:, np.newaxis].reshape(1,4)
            k_mean = pd.DataFrame(k_mean, columns = k_train_for_i.columns)
            
            k_pred_for_i_ = pd.concat((k_pred_for_i_, k_mean))#,ignore_index = True)
        
        k_pred_for_i_.index = k_data_for_i.reset_index()['Time']
        k_pred_for_i_ = k_pred_for_i_.values
        k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...] # batch axis append

        
        
        y_pred_for_i_ = y_train_for_i
        
        for i in range(5):
            data = y_pred_for_i_.iloc[-5  : ]
            y_mean = np.mean(data)
            y_mean = y_mean[:, np.newaxis].reshape(1,1)
            y_mean = pd.DataFrame(y_mean, columns = y_train_for_i.columns)
            
            y_pred_for_i_ = pd.concat((y_pred_for_i_, y_mean))#,ignore_index = True)
        
        y_pred_for_i_.index = y_data_for_i.reset_index()['Time']
        y_pred_for_i_ = y_pred_for_i_.values
        y_pred_for_i_ = y_pred_for_i_[np.newaxis, ...]     

 
        
        #=====================================================================
        
        #이 방법은 안씀...마지막만 5번 반복
        #=====================================================================
        #k_pred_for_i_ = k_train_for_i
        
        #for i in range(5):
         #   k_pred_for_i_ = k_pred_for_i_.append(k_pred_for_i_[-1:])
        
        #k_pred_for_i_.index = k_data_for_i.reset_index()['Time']
        #k_pred_for_i_ = k_pred_for_i_.values
        #k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...] # batch axis append

        
        #y_pred_for_i_ = y_train_for_i
        #for i in range(5):
        #    y_pred_for_i_ = y_pred_for_i_.append(y_pred_for_i_[-1:])
        
        #y_pred_for_i_.index = y_data_for_i.reset_index()['Time']
        #y_pred_for_i_ = y_pred_for_i_.values
        #y_pred_for_i_ = y_pred_for_i_[np.newaxis, ...]     
        
        
        #=====================================================================
        
        
        #랜덤으로
        #=====================================================================
        
        #k_pred_for_i_ = k_train_for_i
        #num = random.randint(0,175)
        #k_pred_for_i_ = k_pred_for_i_.append(k_pred_for_i_[num : num+5])
        #k_pred_for_i_.index = k_data_for_i.reset_index()['Time']
        #k_pred_for_i_ = k_pred_for_i_.values
        #k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...] # batch axis append
        
        #y_pred_for_i_ = y_train_for_i
        #y_pred_for_i_ = y_pred_for_i_.append(y_pred_for_i_[num : num+5])
        #y_pred_for_i_.index = y_data_for_i.reset_index()['Time']
        #y_pred_for_i_ = y_pred_for_i_.values
        #y_pred_for_i_ = y_pred_for_i_[np.newaxis, ...]     
        
        
        #=====================================================================
        
        
        #누적분포
        #=====================================================================
        #k_pred_for_i_ = k_train_for_i
                
        #y_pred_for_i_ = y_train_for_i

                
        #for i in range(5):
        #    u = np.random.uniform(0,100,1)
            
        #    NH4_N = np.percentile(k_pred_for_i_.NH4_N.values, u)
        #    VFA = np.percentile(k_pred_for_i_.VFA.values, u)
        #    ALK = np.percentile(k_pred_for_i_.Alkalinity.values, u)
        #    PH = np.percentile(k_pred_for_i_.PH.values, u)
            
        #    production = np.percentile(y_pred_for_i_.values, u)
            
        #    A = {'NH4_N': NH4_N, 'VFA': VFA, 'Alkalinity': ALK, 'PH': PH}
        #    A = pd.DataFrame(A)
            
        #    B = {'Biogas Production(m3 x 4)':production }
        #    B = pd.DataFrame(B)
            
        #    k_pred_for_i_= k_pred_for_i_.append(A, ignore_index=True)
        #    y_pred_for_i_= y_pred_for_i_.append(B, ignore_index=True)
            
        
        #k_pred_for_i_.index = k_data_for_i.reset_index()['Time']
        #k_pred_for_i_ = k_pred_for_i_.values
        #k_pred_for_i_ = k_pred_for_i_[np.newaxis, ...] 
        
        #y_pred_for_i_.index = y_data_for_i.reset_index()['Time']      
        #y_pred_for_i_ = y_pred_for_i_.values
        #y_pred_for_i_= y_pred_for_i_[np.newaxis, ...]  
        
    
        #=====================================================================
        

        k_pred_for_i_ = k_pred_for_i_[:, -window_size:, :]
        y_pred_for_i_ = y_pred_for_i_[:, -window_size:, :]

        x_pred_for_i = x_data_for_i.iloc[- (window_size ):]            
        kt_pred_for_i = kt_data_for_i.iloc[- (window_size ):]
    
        #print(x_pred_for_i.shape)
        #print(x_pred_for_i[-1:])
        #print(kt_pred_for_i[-1:])
    
        # pred data standardize
        x_pred_mean_for_i = x_pred_for_i.mean()
        x_pred_std_for_i = x_pred_for_i.std()
        
        x_pred_for_i_ = (x_pred_for_i - x_pred_mean_for_i) / (x_pred_std_for_i + 1E-8)
        x_pred_for_i_ = x_pred_for_i_.values.astype(np.float32)[np.newaxis, ...] # batch axis append
     
        kt_pred_mean_for_i = kt_pred_for_i.mean()
        kt_pred_std_for_i = kt_pred_for_i.std()
        
        kt_pred_for_i_ = (kt_pred_for_i - kt_pred_mean_for_i) / (kt_pred_std_for_i + 1E-8)
        kt_pred_for_i_ = kt_pred_for_i_.values.astype(np.float32)[np.newaxis, ...] # batch axis append
            
            
        ########### 원본일때만 죽이고 나머지는 살리기.
        k_pred_mean_for_i = k_pred_for_i_.mean()
        k_pred_std_for_i = k_pred_for_i_.std()
        k_pred_for_i_ = (k_pred_for_i_ - k_pred_mean_for_i) / (k_pred_std_for_i + 1E-8)
            
        y_pred_mean_for_i = y_pred_for_i_.mean()
        y_pred_std_for_i = y_pred_for_i_.std()
        y_pred_for_i_ = (y_pred_for_i_ - y_pred_mean_for_i) / (y_pred_std_for_i + 1E-8)
            
        ###########
            
        pred_k, pred_y = model(x_pred_for_i_, kt_pred_for_i_, k_pred_for_i_, y_pred_for_i_, pinit_states)

        pred_k = pred_k.numpy()
        pred_y = pred_y.numpy()
    
        #print(y_pred_for_i_.shape, pred_y[:, -1, :].shape, (gap + f + 1))
    
        pred_k = pred_k[:, -1, :][np.newaxis, ...]
        pred_y = pred_y[:, -1, :][np.newaxis, ...]
    
        #k_pred_for_i_ = np.concatenate((k_pred_for_i_, pred_k), 1)
        #y_pred_for_i_ = np.concatenate((y_pred_for_i_, pred_y), 1)
        #k_pred_for_i_[:, -1, :] = pred_k
        #y_pred_for_i_ = pred_y
        #y_pred_for_i_[:, -1, :] = pred_y
    
        #print(pred_y[:, -1, :])
    
    #    print(date)
    #    print(y_train_for_i.iloc[-1, :])
    #    print(x_train_for_i.iloc[-1, :])
    #    print(x_pred_for_i.iloc[-1, :])
    
    
    #    pred_y = model(x_pred_for_i_, y_pred_for_i_, pinit_states)
    
    #    pred_y = pred_y.numpy().squeeze()
        #pred_y = pred_y[:, -1, :].squeeze() * y_train_std_for_i + y_train_mean_for_i
        
        pred_k = pred_k.squeeze() * k_train_std_for_i + k_train_mean_for_i
        pred_y = pred_y.squeeze() * y_train_std_for_i + y_train_mean_for_i
        
    
        print('\n\t Prediction result for %s at %s' % (y_data_for_i.index[-1], y_train_for_i.index[-1]))
        
        df_pred_k = pred_k.to_frame()
        #df_pred_k['True'] = true_k
        
        #print(df_pred_k)

        df_pred_y = pred_y.to_frame()
        #df_pred_y['True'] = true_y
        
        #print(df_pred_y)

        pred = pd.concat([df_pred_k, df_pred_y])[0].values

        true = pd.concat([true_k, true_y]).values

    
        true_.append(true)
        pred_.append(pred)
    
    
        mape = np.abs((true - pred) / true)
    
        print('\n\t MAPE')
        print(1. - mape, '\n')
        
        mape_.append(1. - mape)
        
        
mape_
be_loss

print('mean',np.mean(mape_))
#col_mean = pd.DataFrame(np.mean(mape_,axis=0), index=['vfa','alk','vfa/alk','production','new_yield'])
col_mean = pd.DataFrame(np.mean(mape_,axis=0), index=['NH4_N','vfa','alk','PH','production'])
print(col_mean)

'''
#mape 결측값 제외 하고 구하기
start_time = datetime.datetime.strptime(str(metro_intp_data['Time'][-30:].iloc[0]), '%Y-%m-%d %H:%M:%S')
Time = list()  
for i in range((metro_raw_data['Time'] >= start_time).sum()):
            
    num_true = (metro_raw_data['Time'] >= start_time)
    interval_raw = metro_raw_data['Time'][-num_true.sum():]
    f_time = datetime.datetime.strptime(str(interval_raw.iloc[i]), '%Y-%m-%d %H:%M:%S')
    
    interval = abs(int((start_time - f_time).days))

    Time.append(interval)
    

MAPE = list()
for i in range(len(Time)):
    MAPE.append( mape_[Time[i]] )
    
print('mean',np.mean(MAPE))
#col_mean = pd.DataFrame(np.mean(mape_,axis=0), index=['vfa','alk','vfa/alk','production','new_yield'])
col_mean = pd.DataFrame(np.mean(MAPE,axis=0), index=['NH4_N','vfa','alk','PH','production'])
print(col_mean)
'''
#kde 추세값

kde_g = KDE.KDE(df)
jackknife_g = kde_g.b_jackknife(start = 0.1, step = 10) #26.50918061482902
num_bins_g = kde_g.num_bins(jackknife_g, start = 1, stop = 50, step = 10) #11
NW_data_g = kde_g.NW_predict_original(num_bins_g, jackknife_g) #num_bins, b_jackknife
nw_data_g, nw_intp_data_g = kde_g.NW_interpolation(NW_data_g)
df.columns
nw_data_g2 = pd.DataFrame(nw_data_g, columns = ['Time','OLR(kg VS/m3)', 'NH4_N', 'VFA', 'Alkalinity', 'PH', 'Biogas Production(m3 x 4)','Biogas Yield_Vsadd(Nm3/kg Vsadd)']) #나다라야 추세
nw_intp_data_g2 = pd.DataFrame(nw_intp_data_g, columns = ['Time','OLR(kg VS/m3)', 'NH4_N', 'VFA', 'Alkalinity', 'PH', 'Biogas Production(m3 x 4)', 'Biogas Yield_Vsadd(Nm3/kg Vsadd)']) #원데이터 + 나다라야 보간

'''
#산점도 x축,y축
data = df2.values
date = data[:,0]
start_time = datetime.datetime.strptime(str(data[0][0]), '%Y-%m-%d %H:%M:%S')

time = list()
for d in range(len(date)):
    p_time = datetime.datetime.strptime(str(date[d]), '%Y-%m-%d %H:%M:%S')
    time.append(int((p_time - start_time).days) + 1)

Z = time
data_ = data[:,1:]
Y = data_
'''
#####

#예측값 인덱스 설정
g_index = list()
for i in range(189, 189 + test_period):
    g_index.append(i)
    
'NH4_N', 'VFA', 'Alkalinity', 'PH'
    
#예측값
pprod_NH4_H = pd.DataFrame([p[0] for p in pred_], columns = ['NH4_N'])
pprod_vfa = pd.DataFrame([p[1] for p in pred_], columns = ['vfa'])
pprod_alk = pd.DataFrame([p[2] for p in pred_], columns = ['alk'])
pprod_PH = pd.DataFrame([p[3] for p in pred_], columns = ['PH'])
pprod_production = pd.DataFrame([p[4] for p in pred_], columns = ['production'])

pprod_NH4_H.index = g_index
pprod_vfa.index = g_index
pprod_alk.index = g_index
pprod_PH.index = g_index
pprod_production.index = g_index



#################################### 그래프

######## vfa
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))


top.set_xlabel('Time')
top.set_ylabel('VFA')
top.set_title('VFA')

sub1_Left = 189
sub1_Right = 189+30

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[2]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[2]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[2]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_vfa.index,pprod_vfa, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_vfa, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('VFA' )
bottom.set_title('VFA')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[2]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[2]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[2]]], color = 'dodgerblue', label = 'NW')#나다라야 추세
bottom.scatter(pprod_vfa.index,pprod_vfa, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_vfa, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 3200, 8000, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 4000), coordsA=top.transData, 
                       xyB=(sub1_Left, 8000), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)

con2 = patches.ConnectionPatch(xyA=(sub1_Right, 4000), coordsA=top.transData, 
                       xyB=(sub1_Right, 8000 ), coordsB=bottom.transData, color = 'green')

fig.add_artist(con2)
#plt.scatter(Z[:-8], Y[:,2][:-8], 15, color = 'gray', label = 'data') #원데이터 산점도
#plt.plot(Z[:-8], Y[:,2][:-8], color = 'gray')
#plt.plot(df2[:-6][[features[2]]]) #인터폴레이션




#######alk
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('Alkalinity')
top.set_title('Alkalinity')

sub1_Left = 189
sub1_Right = 189+30

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[3]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[3]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[3]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_alk.index,pprod_alk, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_alk, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('Alkalinity' )
bottom.set_title('Alkalinity')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[3]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[3]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[3]]], color = 'dodgerblue', label = 'NW')#나다라야 추세
bottom.scatter(pprod_alk.index,pprod_alk, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_alk, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)



bottom.fill_between((sub1_Left,sub1_Right), 8000, 15000, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 8000), coordsA=top.transData, 
                       xyB=(sub1_Left, 15000), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 8000), coordsA=top.transData, 
                       xyB=(sub1_Right, 15000), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)


#plt.scatter(Z[:-8], Y[:,3][:-8], 15, color = 'grey', label = 'data') #원데이터 산점도
#plt.plot(Z[:-8], Y[:,3][:-8], color = 'gray')
#plt.plot(df2[:-6][[features[3]]]) #인터폴레이션





###################  NH4
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('NH4_N')
top.set_title('NH4_N')

sub1_Left = 189
sub1_Right = 189+30
#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[1]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[1]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[1]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_NH4_H.index, pprod_NH4_H, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_NH4_H, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('NH4_N' )
bottom.set_title('NH4_N')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[1]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[1]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[1]]], color = 'dodgerblue' , label = 'NW')#나다라야 추세
bottom.scatter(pprod_NH4_H.index, pprod_NH4_H, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_NH4_H, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 1800, 3300, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 1800), coordsA=top.transData, 
                       xyB=(sub1_Left, 3300), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 1800), coordsA=top.transData, 
                       xyB=(sub1_Right, 3300), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)






################PH
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('PH')
top.set_title('PH')

sub1_Left = 189
sub1_Right = 189+30
#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[4]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[4]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[4]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_PH.index, pprod_PH, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_PH, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('PH' )
bottom.set_title('PH')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[4]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[4]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[4]]], color = 'dodgerblue', label = 'NW')#나다라야 추세
bottom.scatter(pprod_PH.index, pprod_PH, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_PH, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 7.4, 7.85, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left,7.4), coordsA=top.transData, 
                       xyB=(sub1_Left, 7.85), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 7.4), coordsA=top.transData, 
                       xyB=(sub1_Right, 7.85), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)



#############production
plt.style.use('seaborn')
fig = plt.figure(figsize=(18, 10))
top = fig.add_subplot(5,10,(5,19))

top.set_xlabel('Time')
top.set_ylabel('Biogas Production(m3 x 4)')
top.set_title('Biogas Production(m3 x 4)')

sub1_Left = 189
sub1_Right = 189+30

#top
top.scatter(nw_intp_data_g2.index[:-6][sub1_Left:sub1_Right], nw_intp_data_g2[:-6][features[5]][sub1_Left:sub1_Right], 15 ,color = 'gray') #나다라야 보간값 산점도
top.plot(nw_intp_data_g2[:-6][features[5]][sub1_Left:sub1_Right], color = 'gray') #나다라야 보간값 선
top.plot(nw_data_g2[:-6][[features[5]]][sub1_Left:sub1_Right], color = 'dodgerblue')#나다라야 추세
top.scatter(pprod_production.index, pprod_production, 10 ,color = 'red') # 예측값 산점도
top.plot(pprod_production, color = 'red') #예측값 선

#bottom
bottom = fig.add_subplot(2,2,(3,4))
bottom.set_xlabel('Time' )
bottom.set_ylabel('Biogas Production(m3 x 4)' )
bottom.set_title('Biogas Production(m3 x 4)')
bottom.scatter(nw_intp_data_g2.index[:-6], nw_intp_data_g2[:-6][features[5]], 15 ,color = 'gray') #나다라야 보간값 산점도
bottom.plot(nw_intp_data_g2[:-6][features[5]], color = 'gray', label = 'Interpolation') #나다라야 보간값 선
bottom.plot(nw_data_g2[:-6][[features[5]]], color = 'dodgerblue' , label='NW')#나다라야 추세
bottom.scatter(pprod_production.index, pprod_production, 10 ,color = 'red') # 예측값 산점도
bottom.plot(pprod_production, color = 'red', label = 'pred') #예측값 선
bottom.legend(loc= 'upper left', fontsize=14,frameon=True,shadow=True)


bottom.fill_between((sub1_Left,sub1_Right), 30000, 52000, facecolor='green', alpha=0.2)
con1 = patches.ConnectionPatch(xyA=(sub1_Left, 30000), coordsA=top.transData, 
                       xyB=(sub1_Left, 52000), coordsB=bottom.transData, color = 'green')
fig.add_artist(con1)
con2 = patches.ConnectionPatch(xyA=(sub1_Right, 30000), coordsA=top.transData, 
                       xyB=(sub1_Right, 52000), coordsB=bottom.transData, color = 'green')
fig.add_artist(con2)




'''
#yield
plt.figure(figsize=(22, 5))
plt.style.use('seaborn')
plt.xlabel('Time',fontsize=17)
plt.ylabel('Biogas Yield_Vsadd(Nm3/kg Vsadd)',fontsize=17)
plt.title('Biogas Yield_Vsadd(Nm3/kg Vsadd)',fontsize=22)
plt.scatter(Z[:-8], Y[:,6][:-8], 15, color = 'grey', label = 'data') #원데이터 산점도
plt.plot(Z[:-8], Y[:,6][:-8], color = 'gray')
#plt.plot(df2[:-6][[features[6]]]) #인터폴레이션
plt.plot(nw_data_g2[:-6][[features[6]]],color = 'dodgerblue')#나다라야 추세
plt.plot(pprod_yield,color = 'red')

#new_yield
plt.figure(figsize=(22, 5))
plt.style.use('seaborn')
plt.xlabel('Time',fontsize=17)
plt.ylabel('new_yield',fontsize=17)
plt.title('new_yield',fontsize=22)
plt.scatter(Z[:-8], Y[:,8][:-8], 15, color = 'grey', label = 'data') #원데이터 산점도
plt.plot(Z[:-8], Y[:,8][:-8], color = 'gray')
#plt.plot(df2[:-6][[features[8]]]) #인터폴레이션
plt.plot(nw_data_g2[:-6][[features[8]]],color = 'dodgerblue')#나다라야 추세
plt.plot(pprod_new_yield,color = 'red')
'''











