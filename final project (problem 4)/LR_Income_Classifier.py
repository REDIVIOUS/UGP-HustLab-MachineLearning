import numpy as np
from matplotlib import pyplot as plt
import csv

# 读取文件数据
def ReadData():
    origin_data = []
    with open('data.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            origin_data.append(row)

    data = np.array(origin_data)

    Train_X = np.zeros((3000,57))
    Train_y = np.zeros((3000,1))
    Test_X = np.zeros((1000,57))
    Test_y = np.zeros((1000,1))

    # 从csv读进来是文本形式，现在转变为float浮点数形式，方便后续计算
    for i in range(0,3000):
        for j in range(1,58):
            Train_X[i][j-1] = float(data[i][j])
        Train_y[i][0] = float(data[i][58])
    for i in range(3000,4000):
        for j in range(1,58):
            Test_X[i-3000][j-1] = float(data[i][j])
        Test_y[i-3000][0] = float(data[i][58])

    # 返回训练数据、训练标签、测试数据、测试标签
    return Train_X,Train_y,Test_X,Test_y

# 均值归一化（正规化），提升模型准确性，提高收敛速度
def Normalization(x):
    u = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x - u) / sigma
    return x


# sigmoid函数
# 这里我们需要用clip函数规定最大最小值
# 其原因是，我们后续要将函数值串入log函数中
# 如果太小，会超出计算机计算范围，使得程序数据出错（反复实验得出的结论）
def sigmoid(z):
    sig = 1.0 / (1.0 + np.exp(-z))
    return np.clip(sig, 1e-8, 1-(1e-8))

# 损失函数（对数似然+正则项)
# 这里的代价函数采用对数似然以及正则项，其中lamda为正则化参数
# 由于下面我们要用梯度下降求，在对数似然加上常量-1/m可以满足条件
# Loss = -1/m((yi * log(h(xi)) + (1-yi) * log(1 - h(i)) + lamda/2m * w^2
def LossReg_function(X,y,z,w,lamda):
    sig = sigmoid(z)
    A = y * np.log(sig)
    B = (1 - y) * np.log(1 - sig)
    # 正则项从1开始到n求和
    reg_exp = (lamda/(2 * len(X))) * np.sum(np.power(w,2))
    return (-np.sum(A + B)) / len(X) + reg_exp


# 梯度下降过程 
def GradientDecent(X,y,w,b,lamda,alpha,times):
    LossList = []
    for i in range(times):
        XData = np.array(X) 
        yData = np.array(y)

        # 计算点的位置z
        z = np.dot(XData,w) + b
        # 计算真实值和估计值的差距
        ErrorLoss = sigmoid(z) - yData

        # 对w和b梯度下降
        grad_w = np.dot(XData.transpose(),ErrorLoss) / len(XData) + (lamda/len(XData)) * w # 求偏导数
        grad_b = np.sum(ErrorLoss) / len(XData)

        # 下降过程
        w = w - alpha * grad_w
        b = b - alpha * grad_b
        
        # 计算本步损失
        z = np.dot(XData,w) + b
        LossInfo = LossReg_function(XData,yData,z,w,lamda)
        LossList.append(LossInfo) #迭代过程中加入每一步的loss
        
    return w, b , LossList

# 计算收敛时间
def calclose(LossList):
    # 计算收敛的时间
    # 我们认为这一步和上一步的相对误差在10-7之内，就算做已经收敛
    for i in range(len(LossList)):
        E = abs(LossList[i] - LossList[i - 1]) / LossList[i - 1]
        if(E < 1e-6):
            return i
    return "none"
        

# 预测模块，预测大于0.5即为标签为1，小雨0.5即标签为0
def predict(Test_X, Test_y ,w_result,b_result):
    count = 0
    for i in range(0,1000):
        z = np.dot(Test_X[i],w_result) + b_result
        k = sigmoid(z)
        if (k[0] >= 0.5):
            if(Test_y[i][0] == 1): 
                count = count + 1 # 预测大于0.5且真实值为1的可以算正确
        elif (k[0] < 0.5):
            if(Test_y[i][0] == 0):
                count = count + 1 # 预测小于0.5且真实值为0的可以算正确
    acc = count/1000.0
    mis = 1.0 - acc
    return mis

# 迭代一次，预测一次模块，用于绘制misclassification曲线，返回错误率表
def PredictGradient(X,y,w,b,lamda,alpha,times,Test_X,Test_y):
    mis = [] #错误率列表
    for i in range(times):
        XData = np.array(X)
        yData = np.array(y)

        # 计算点的位置z
        z = np.dot(XData,w) + b
        # 计算真实值和估计值的差距
        ErrorLoss = sigmoid(z) - yData

        # 对w和b梯度下降
        grad_w = np.dot(XData.transpose(),ErrorLoss) / len(XData) - 2 * lamda * w /len(XData) # 求偏导数
        grad_b = np.sum(ErrorLoss) / len(XData)

        # 下降过程
        w = w - alpha * grad_w
        b = b - alpha * grad_b
        
        mis.append(predict(Test_X,Test_y,w,b))
        print("one,finished!",i)
    return mis



if __name__ == '__main__':
    # 参数确定
    lamda = 100 # 正则化参数的确定
    alpha =  0.001 # 学习率的确定
    times = 1000000 # 梯度下降迭代次数

    # 读取数据
    Train_X_origin,Train_y,Test_X_origin,Test_y = ReadData()
    Train_X = Normalization(Train_X_origin) 
    Test_X = Normalization(Test_X_origin)

    # w、b初始化
    w = np.full((57,1),0)
    b = 0

    # 探究不同的参数对性能的影响
    # 改变学习率
    # 我们迭代100000次，分别选择几个常见的学习率：0.1、0.01和0.001的的曲线，观察loss值的收敛情况
    w_result_1, b_result_1 ,Loss_result_1 = GradientDecent(Train_X,Train_y,w,b,0,0.1,100000)
    print("alpha = 0.1: 收敛时间：", calclose(Loss_result_1))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_1,b_result_1))
    w_result_2, b_result_2 ,Loss_result_2 = GradientDecent(Train_X,Train_y,w,b,0,0.01,100000)
    print("alpha = 0.01: 收敛时间：", calclose(Loss_result_2))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_2 ,b_result_2))
    w_result_3, b_result_3 ,Loss_result_3 = GradientDecent(Train_X,Train_y,w,b,0,0.001,100000)
    print("alpha = 0.001: 收敛时间：", calclose(Loss_result_3))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_3,b_result_3))

    plt.title('iter: 100000, cost of loss function')
    plt.plot(Loss_result_1, color='green', label='alpha = 0.1')
    plt.plot(Loss_result_2, color='red', label='alpha = 0.01')
    plt.plot(Loss_result_3, color='blue', label='alpha = 0.001')
    plt.xlabel('iter times')
    plt.ylabel('value of loss function')
    plt.legend() # 显示图例
    plt.show()

    # 实验过程中发现，如果学习率选择过小，会产生震荡的情况
    # 在上面的情况下，继续增加学习率，选择50、5、0.5三个学习率
    w_result_4, b_result_4 ,Loss_result_4 = GradientDecent(Train_X,Train_y,w,b,0,50,100000)
    print("alpha = 50: 收敛时间：", calclose(Loss_result_4))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_4,b_result_4))
    w_result_5, b_result_5, Loss_result_5 = GradientDecent(Train_X,Train_y,w,b,0,5,100000)
    print("alpha = 5: 收敛时间：", calclose(Loss_result_5))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_5,b_result_5))
    w_result_6, b_result_6 ,Loss_result_6 = GradientDecent(Train_X,Train_y,w,b,0,0.5,100000)
    print("alpha = 0.5: 收敛时间：", calclose(Loss_result_6))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_6,b_result_6))

    plt.title('iter: 100000, cost of loss function')
    plt.plot(Loss_result_4, color='green', label='alpha = 50')
    plt.plot(Loss_result_5, color='red', label='alpha = 5')
    plt.plot(Loss_result_6, color='blue', label='alpha = 0.5')
    plt.xlabel('iter times')
    plt.ylabel('value of loss function')
    plt.legend() # 显示图例
    plt.show()

    # 探究学习率以及迭代次数对预测的影响
    # 根据经验取四个合适的学习率，分别是1、0.1、0.01、0.001，我们将分别对其迭代1000000次
    # 画出每个学习率随着迭代次数的misclassification的图线
    result_1 = PredictGradient(Train_X,Train_y,w,b,0,1,1000,Test_X,Test_y)
    result_2 = PredictGradient(Train_X,Train_y,w,b,0,0.1,15000,Test_X,Test_y)
    result_3 = PredictGradient(Train_X,Train_y,w,b,0,0.01,60000,Test_X,Test_y)
    result_4 = PredictGradient(Train_X,Train_y,w,b,0,0.001,300000,Test_X,Test_y)

    plt.title('iter: 1000, misclassification rate')
    plt.plot(result_1, color='purple', label='alpha = 1')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 15000, misclassification rate')
    plt.plot(result_2, color='dodgerblue', label='alpha = 0.1')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 60000, misclassification rate')
    plt.plot(result_3, color='orangered', label='alpha = 0.01')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 300000, misclassification rate')
    plt.plot(result_4, color='green', label='alpha = 0.001')
    plt.legend() # 显示图例
    plt.show()
    
    # 探究正则系数对实验的影响
    w_result_7_1, b_result_7_1 ,Loss_result_7_1 = GradientDecent(Train_X,Train_y,w,b,0,0.1,100000)
    print("alpha = 0.1、lamda = 0: 收敛时间：", calclose(Loss_result_7_1))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_7_1,b_result_7_1))
    w_result_7_2, b_result_7_2 ,Loss_result_7_2 = GradientDecent(Train_X,Train_y,w,b,10,0.1,100000)
    print("alpha = 0.1、lamda = 10: 收敛时间：", calclose(Loss_result_7_2))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_7_2,b_result_7_2))
    w_result_7_3, b_result_7_3 ,Loss_result_7_3 = GradientDecent(Train_X,Train_y,w,b,100,0.1,100000)
    print("alpha = 0.1、lamda = 100: 收敛时间：", calclose(Loss_result_7_3))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_7_3,b_result_7_3))
    w_result_7_4, b_result_7_4 ,Loss_result_7_4 = GradientDecent(Train_X,Train_y,w,b,1000,0.1,100000)
    print("alpha = 0.1、lamda = 1000: 收敛时间：", calclose(Loss_result_7_4))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_7_4,b_result_7_4))


    w_result_8_1, b_result_8_1, Loss_result_8_1 = GradientDecent(Train_X,Train_y,w,b,0,0.01,100000)
    print("alpha = 0.01、lamda = 0: 收敛时间：", calclose(Loss_result_8_1))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_8_1,b_result_8_1))
    w_result_8_2, b_result_8_2, Loss_result_8_2 = GradientDecent(Train_X,Train_y,w,b,10,0.01,100000)
    print("alpha = 0.01、lamda = 10: 收敛时间：", calclose(Loss_result_8_2))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_8_2,b_result_8_2))
    w_result_8_3, b_result_8_3, Loss_result_8_3 = GradientDecent(Train_X,Train_y,w,b,100,0.01,100000)
    print("alpha = 0.01、lamda = 100: 收敛时间：", calclose(Loss_result_8_3))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_8_3,b_result_8_3))
    w_result_8_4, b_result_8_4, Loss_result_8_4 = GradientDecent(Train_X,Train_y,w,b,1000,0.01,100000)
    print("alpha = 0.01、lamda = 1000: 收敛时间：", calclose(Loss_result_8_4))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_8_4,b_result_8_4))



    w_result_9_1, b_result_9_1 ,Loss_result_9_1 = GradientDecent(Train_X,Train_y,w,b,0,0.001,100000)
    print("alpha = 0.001、lamda = 0: 收敛时间：", calclose(Loss_result_9_1))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_9_1,b_result_9_1))
    w_result_9_2, b_result_9_2 ,Loss_result_9_2 = GradientDecent(Train_X,Train_y,w,b,10,0.001,100000)
    print("alpha = 0.001、lamda = 10: 收敛时间：", calclose(Loss_result_9_2))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_9_2,b_result_9_2))
    w_result_9_3, b_result_9_3 ,Loss_result_9_3 = GradientDecent(Train_X,Train_y,w,b,100,0.001,100000)
    print("alpha = 0.001、lamda = 100: 收敛时间：", calclose(Loss_result_9_3))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_9_3,b_result_9_3))
    w_result_9_4, b_result_9_4 ,Loss_result_9_4 = GradientDecent(Train_X,Train_y,w,b,1000,0.001,100000)
    print("alpha = 0.001、lamda = 1000: 收敛时间：", calclose(Loss_result_9_4))
    print("misclassification-rate: ", predict(Test_X,Test_y,w_result_9_4,b_result_9_4))


    # 探究没有正规化的时候的收敛情况和正确率
    w_result_10, b_result_10 ,Loss_result_10 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.1,100000)
    print("alpha = 0.1:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_10,b_result_10))
    w_result_11, b_result_11 ,Loss_result_11 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.01,100000)
    print("alpha = 0.01:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_11 ,b_result_11))
    w_result_12, b_result_12 ,Loss_result_12 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.001,100000)
    print("alpha = 0.001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_12 ,b_result_12))
    w_result_13, b_result_13 ,Loss_result_13 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.0001,100000)
    print("alpha = 0.0001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_13,b_result_13))
    w_result_14, b_result_14 ,Loss_result_14 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.00001,100000)
    print("alpha = 0.00001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_14,b_result_14))
    w_result_15, b_result_15 ,Loss_result_15 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.000001,100000)
    print("alpha = 0.000001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_15,b_result_15))
    w_result_16, b_result_16 ,Loss_result_16 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.0000001,100000)
    print("alpha = 0.0000001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_16,b_result_16))
    w_result_17, b_result_17 ,Loss_result_17 = GradientDecent(Train_X_origin,Train_y,w,b,0,0.00000001,100000)
    print("alpha = 0.00000001:")
    print("misclassification-rate: ", predict(Test_X_origin,Test_y,w_result_17,b_result_17))
    
    plt.title('iter: 100000, cost of learning rate of 0.1, 0.01')
    plt.plot(Loss_result_10, color='green', label='alpha = 0.1')
    plt.plot(Loss_result_11, color='red', label='alpha = 0.01')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 100000, cost of learning rate of 0.001, 0.0001')
    plt.plot(Loss_result_12, color='blue', label='alpha = 0.001')
    plt.plot(Loss_result_13, color='black', label='alpha = 0.0001')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 100000, cost of learning rate of 0.00001, 0.000001')
    plt.plot(Loss_result_14, color='yellow', label='alpha = 0.00001')
    plt.plot(Loss_result_15, color='purple', label='alpha = 0.000001')
    plt.legend() # 显示图例
    plt.show()
    plt.title('iter: 100000, cost of learning rate of 0.0000001, 0.00000001')
    plt.plot(Loss_result_16, color='blue', label='alpha = 0.0000001')
    plt.plot(Loss_result_17, color='green', label='alpha = 0.00000001')
    plt.legend() # 显示图例
    plt.show()

