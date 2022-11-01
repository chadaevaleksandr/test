import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


#загружаю данные из файла
data = genfromtxt('train.csv', delimiter=',')
data = data[1:]
print('прочитано ', data.shape)
np.random.shuffle(data)

def data_saturate(data, new_percent):
    # равномерно распределяю по массиву значения с единицами (не использую)
    # увеличиваю процент значений с расширившимся спредом до заданного (не использую)
    data_0 = np.array([[]])
    data_1 = np.array([[]])
    cnt0 = 0
    cnt1 = 0
    for i in range(0, len(data)):
        if data[i][-1] == 0:
            if cnt0 == 0:
                data_0 = np.concatenate((data_0, [data[i]]), axis=1)
            else:
                data_0 = np.concatenate((data_0, [data[i]]), axis=0)
            cnt0 += 1
        else:
            if cnt1 == 0:
                data_1 = np.concatenate((data_1, [data[i]]), axis=1)
            else:
                data_1 = np.concatenate((data_1, [data[i]]), axis=0)
            cnt1 += 1

    #print('из них:')
    #print(data_0.shape, ' # нулевых значений')
    print(data_1.shape, ' # единиц')

    if new_percent > 0:
        new_len_0 = int(round(len(data_1) * (100 - new_percent) / new_percent, 0))
        if new_len_0 < len(data_0):
            data_0 = data_0[:new_len_0]
        #print(data_0.shape, ' # новое количество значений')

    d0_len = len(data_0)
    d1_len = len(data_1)
    freq = (d0_len + d1_len)/d1_len
    newData = np.zeros((len(data), len(data[0])))
    posArr = []
    for i in range(0, d1_len):
        posArr.append(int(round((i + 1)*freq - 1, 0)))
    cnt0 = 0
    for i in range(0, len(newData)):
        if i in posArr:
            newData[i] = data_1[posArr.index(i)]
        else:
            newData[i] = data_0[cnt0]
            cnt0 += 1

    #newData = np.concatenate((data_0, data_1), axis=0)
    #np.random.shuffle(newData)
    print(newData.shape, ' # массив для работы')
    return newData

def show_rez(history):
    # вывод истории обучения модели графиком
    fig, axs = plt.subplots(2)
    fig.set_size_inches(7, 10)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    axs[0].plot(epochs, loss, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
    axs[0].legend()

    loss = history.history['auc']
    val_loss = history.history['val_auc']
    epochs = range(1, len(loss) + 1)
    axs[1].plot(epochs, loss, 'bo', label='Training auc')
    axs[1].plot(epochs, val_loss, 'b', label='Validation auc')
    axs[1].legend()

    plt.show()
'''
data = data_saturate(data, 0)
np.save('prep_data', data)
data = np.load('prep_data.npy')
'''
targets = data[:, -1:]
data = data[:, :-1]

#тут можно попробовать донасыщать данные признаками

#формирую массивы train, test с данными и метками(целями)
len_data = len(data)
train_perc = int(round(0.75*len_data, 0))
test_perc1 = int(round(len_data, 0))

train_data = data[:train_perc]
train_targets = targets[:train_perc]
test_data = data[train_perc:test_perc1]
test_targets = targets[train_perc:test_perc1]

#подготавливаю массивы чтобы скормить нейронке
print('Форма данных на входе:')
print(train_data.shape)
print(test_data.shape)

#размещаю данные за прошлый период под текущим, чтобы разделить на std
train_data = np.reshape(train_data, (len(train_data) * 2, 10))
test_data = np.reshape(test_data, (len(test_data) * 2, 10))
std = train_data.std(axis=0)
train_data /= std
test_data /= std

#формирую массив для рекуррентной сети
train_data = np.reshape(train_data, (int(round(len(train_data)/2, 0)), 2, 10))
for i in range(0, len(train_data)):
    train_data[i][[0, 1]] = train_data[i][[1, 0]]
test_data = np.reshape(test_data, (int(round(len(test_data)/2, 0)), 2, 10))
for i in range(0, len(test_data)):
    test_data[i][[0, 1]] = test_data[i][[1, 0]]

#оставляю в массиве только нужные столбцы
new_columns = np.zeros((len(train_data), 2, 4))
for i in range(0, len(train_data)):
    new_columns[i][0] = np.array([train_data[i][0][0], train_data[i][0][1], train_data[i][0][5], train_data[i][0][6]])
    new_columns[i][1] = np.array([train_data[i][1][0], train_data[i][1][1], train_data[i][1][5], train_data[i][1][6]])
train_data = new_columns
new_columns = np.zeros((len(test_data), 2, 4))
for i in range(0, len(test_data)):
    new_columns[i][0] = np.array([test_data[i][0][0], test_data[i][0][1], test_data[i][0][5], test_data[i][0][6]])
    new_columns[i][1] = np.array([test_data[i][1][0], test_data[i][1][1], test_data[i][1][5], test_data[i][1][6]])
test_data = new_columns

print('Форма данных для загрузки в нейросеть:')
print(train_data.shape)
print(test_data.shape)



model = models.Sequential()
model.add(layers.LSTM(32, input_shape=(None, train_data.shape[-1])))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005), loss='binary_crossentropy', metrics=['AUC'])
model.summary()
history = model.fit(train_data, train_targets, epochs=400, shuffle=True, batch_size=900, verbose=1, validation_data=(test_data, test_targets))
#model.save('model.h5')




#from keras.models import load_model
#model = load_model('model.h5')

#загружаю и подготавливаю данные для прогнозирования
pred_data = genfromtxt('test.csv', delimiter=',')
pred_data = pred_data[1:]

# размещаю данные за прошлый период под текущим, чтобы разделить на std
pred_data = np.reshape(pred_data, (len(pred_data) * 2, 10))
pred_data /= std

# формирую массив для рекуррентной сети
pred_data = np.reshape(pred_data, (int(round(len(pred_data) / 2, 0)), 2, 10))
for i in range(0, len(pred_data)):
    pred_data[i][[0, 1]] = pred_data[i][[1, 0]]

# оставляю в массиве только нужные столбцы
new_columns = np.zeros((len(pred_data), 2, 4))
for i in range(0, len(pred_data)):
    new_columns[i][0] = np.array([pred_data[i][0][0], pred_data[i][0][1], pred_data[i][0][5], pred_data[i][0][6]])
    new_columns[i][1] = np.array([pred_data[i][1][0], pred_data[i][1][1], pred_data[i][1][5], pred_data[i][1][6]])
pred_data = new_columns


predictions = model.predict(pred_data)
np.savetxt('prediction.csv', predictions, header='Y_PRED', delimiter = ',', fmt = '%1.5f', comments='')


show_rez(history)