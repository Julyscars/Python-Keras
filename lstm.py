import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#%matplotlib inline

if __name__ == '__main__':
    # load the dataset
    dataframe = read_csv('qunzu.csv', usecols=[1], engine='python', skipfooter=0)
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    plt.xlabel('day')
    plt.ylabel('number of error logs')
    plt.plot(dataset)
    #plt.show()

    # X is the number of logs at a given time (t)
    # Y is the number of logs at the next time (t + 1).

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        #print(numpy.array(dataX))
        #print(numpy.array(dataY))
        return numpy.array(dataX), numpy.array(dataY)

    # fix random seed for reproducibility
    numpy.random.seed(7)

    # normalize the dataset
    # 去掉量纲
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)


    # split into train and test sets
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # use this function to prepare the train and test datasets for modeling
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #print("testX:",testX)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2,validation_data=(testX,testY))

    # make predictions

    trainPredict = model.predict(trainX)
    #print("trainPredict",trainPredict)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)

    trainY = scaler.inverse_transform([trainY])

    testPredict = scaler.inverse_transform(testPredict)

    testY = scaler.inverse_transform([testY])

  #  print("trainPredict", trainPredict)
  #  print("trainY", trainY)
  #  print("testPredict", testPredict)
  #  print("testY",testY)

    # 增益处理
    # 训练集预测增益处理
    trP_mean = numpy.mean(trainPredict[:, 0])
    #print("trP_mean:",trP_mean)
    trP_median = numpy.median(trainPredict[:, 0])
    #print("trP_median:", trP_median)
    #print(str(numpy.mean(trainY[0])) + ':' + str(trainY[0].max()))
    trainPredict[:, 0] = (trainPredict[:, 0]-trP_mean)*1.5+trP_mean
    #print(str(numpy.mean(trainY[0]))+':'+str(trainY[0].max()))
    print("trainPredict[:, 0]:",trainPredict[:, 0])
    # 测试集预测增益处理
    teP_mean = numpy.mean(testPredict[:, 0])
    teP_median = numpy.median(testPredict[:, 0])
    #print(str(numpy.mean(trainPredict[:, 0])) + ':' + str(trainPredict[:, 0].max()))
    testPredict[:, 0] = (testPredict[:, 0]-teP_mean)*1.5+teP_mean
    #print(str(numpy.mean(trainPredict[:, 0])) + ':' + str(trainPredict[:, 0].max()))
    print("testPredict[:, 0]:",testPredict[:, 0])
    # 训练集对齐
    trY = numpy.delete(trainY[0], len(trainY[0]) - 1, 0)
    trP = numpy.delete(trainPredict[:, 0], 0, 0)

    # 均方根误差
    trainScore = math.sqrt(mean_squared_error(trY, trP))
    print('Train Score: %.2f RMSE' % (trainScore))

    # 测试集对齐
    teY = numpy.delete(testY[0], len(testY[0]) - 1, 0)
    teP = numpy.delete(testPredict[:, 0], 0, 0)
    testScore = math.sqrt(mean_squared_error(teY, teP))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    original_log = scaler.inverse_transform(dataset)
    #print(len(trainPredict)+len(testPredict))
    print('original_stdvar:%.2f'%numpy.sqrt(original_log.var()))
    print('trainPredict_stdvar:%.2f'%numpy.sqrt(trainPredict.var()))
    print('testPredict_stdvar:%.2f'%numpy.sqrt(testPredict.var()))
    trainPredictPlot = numpy.delete(trainPredictPlot,0,0)
    testPredictPlot = numpy.delete(testPredictPlot,0,0)
    plt.plot(scaler.inverse_transform(dataset),color='blue',label='ground truth')
    plt.plot(trainPredictPlot,color='orange',label='train predicting',linestyle='--')
    plt.plot(testPredictPlot,color='green',label='test predicting',linestyle='-.')
    plt.legend()
    plt.show()
