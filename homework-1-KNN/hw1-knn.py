import numpy as np
import struct
from matplotlib import pyplot as plt
import seaborn as sns

# knn算法的实现
def knn(train_data, test_data, train_label, k):
    n = train_data.shape[0] # n代表训练样本的个数（训练集的行数）
    test_array = np.tile(test_data, (n,1))
    array_diff = test_array - train_data # 求横纵坐标差
    distance = ((array_diff**2).sum(axis=1))**0.5 # 求test_data到每个点的train_data的每个点距离
    SortedDisIndex = distance.argsort() # 返回排序好了之后距离的数组索引
    # ShowKnnImage(train_data, test_data, train_label, k ,SortedDisIndex) # 需要输出k临近图片时候解注释
    counts = {} # 记录每一个label的数量
    for i in range(k): # 记录距离前k个点所属的标签类别
        label = train_label[SortedDisIndex[i]]
        counts[label] = counts.get(label,0)+1 # 将对应类别创建并加一
    # 求出种类最多的label
    counts_list = sorted(counts.items(), key = lambda x:x[1],reverse = True) # 降序排列
    return counts_list[0][0] #返回最多的种类的label

# 显示k近邻图片模块
def ShowKnnImage(train_data, test_data, train_label, k ,SortedDisIndex):
    # 显示测试图片
    img = test_data.reshape(28,28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title("label: %d" %test_label[5])
    plt.show()
    # 显示k临近图片
    fig,ax = plt.subplots(ncols = 4, nrows = 4,sharex = True, sharey =True)
    axes = ax.flatten()
    for i in range(k):
        img = train_data[SortedDisIndex[i]].reshape(28,28)
        axes[i].imshow(img, cmap='Greys', interpolation='nearest')
        axes[i].set_title("label: %d" %train_label[SortedDisIndex[i]])
    plt.show()

# 读取图像
# 图像前16字节是头，以4字节为单位分别表示无意义数、图像数量、单个图像行数、单个图像列数
def ReadMnistImage(data_file):
    content = open(data_file,"rb").read() # 文件内容
    head = struct.unpack_from('>IIII', content, 0) # 读取文件头
    # 读取图片数量、图片宽度、图片高度
    num = head[1]
    row = head[2]
    column = head[3]
    ImageSize = row * column # 计算图片大小
    image = np.empty((num, 28 * 28))# 初始化图片矩阵，先全部为0(图片在28*28的中心)
    fmt = '>' + str(ImageSize) + 'B' # 读取单个图片的格式
    offset = struct.calcsize('>IIII') # 头文件的偏移
    # 循环读取文件的内容（以一张图片为单位），用numpy变成array形式
    for i in range(num):
        image [i] = np.array(struct.unpack_from(fmt, content, offset))
        offset += struct.calcsize(fmt)
    return image

# 读取标记
# 标记前8字节是头，以4字节为单位分别表示无意义数、标记的数量
def ReadMnistLabel(label_file):
    content = open(label_file,"rb").read() # 文件内容
    head = struct.unpack_from('>II', content, 0) # 读取文件头
    num = head[1] # 标记数量
    fmt = '>' + str(num) + 'B' # 读取label的格式
    offset = struct.calcsize('>II') # 读取偏移（可视作文件指针）
    label = np.array(struct.unpack_from(fmt, content, offset)) # 读取label内容，用numpy转换为array形式
    return label

# 调用knn算法计算miscalssification_rate
def TrainKnn(k, train_data, train_lable, train_num ,test_data, test_label ,test_num):
    # 指定特定大小的训练集，取出对应的data和label
    train_data_use = np.empty((train_num, 28*28)) # 实际使用的训练集图像
    train_lable_use = [] # 实际使用的训练集标签
    for i in range(train_num):
        train_data_use[i] = train_data[i]
        train_lable_use.append(train_label[i])

    # 调用knn训练模块
    Error = 0 # 错误的个数
    # 测试的数据前一半从前5000张测试集（好辨识的）中取出
    half_test_num = int(test_num/2)
    for i in range(half_test_num):
        CalLabel = knn(train_data_use, test_data[i], train_lable_use, k)
        if CalLabel != test_label[i]:
            Error += 1.0
    # 测试的数据后一半从后5000张测试集（不好辨识的）中取出
    for i in range(half_test_num):
        CalLabel = knn(train_data_use, test_data[i + 5000], train_lable_use, k)
        if CalLabel != test_label[i + 5000]:
            Error += 1.0
    miscalssification_rate = Error/test_num # 错误率
    print("当前k值: ", k)
    print("测试总数: ", test_num)
    print("错误个数: ", Error)
    print("错误率: ", miscalssification_rate)
    print("正确率: ", 1 - miscalssification_rate)
    print("\n")
    return miscalssification_rate


if __name__ == '__main__':
    # 读取文件中的数据，转成可以被处理的矩阵形式
    train_data = ReadMnistImage("train-images.idx3-ubyte")
    train_label = ReadMnistLabel("train-labels.idx1-ubyte")
    test_data = ReadMnistImage("t10k-images.idx3-ubyte")
    test_label = ReadMnistLabel("t10k-labels.idx1-ubyte")

    # 参数表
    k = 15
    test_num = 1000
    # 训练集3000的时候，调用knn计算miscalssification_rate
    Rate_List_3000 = []
    for i in range(1, k + 1):
        result = TrainKnn(i, train_data, train_label, 3000, test_data, test_label, test_num)
        Rate_List_3000.append(result)

    # # 训练集为10000的时候，调用knn计算miscalssification_rate
    Rate_List_10000 = []
    for i in range(1, k + 1):
        result = TrainKnn(i, train_data, train_label, 10000, test_data, test_label, test_num)
        Rate_List_10000.append(result)
    
    # # 训练集为60000的时候，调用knn计算miscalssification_rate
    Rate_List_60000 = []
    for i in range(1, k + 1):
        result = TrainKnn(i, train_data, train_label, 60000, test_data, test_label, test_num)
        Rate_List_60000.append(result)


    # 可视化
    plt.title('testing set: 1000, miscalssification_rate changes with k')
    plt.plot(Rate_List_3000, color='green', label='training set: 3000')
    plt.plot(Rate_List_10000, color='blue', label='training set: 10000')
    plt.plot(Rate_List_60000, color='red', label='training set :60000')
    plt.legend() # 显示图例
 
    plt.xlabel("value of k")
    plt.ylabel("miscalssification_rate")
    plt.show()





