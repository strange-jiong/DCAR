import numpy as np
import sys


class DataSet(object):
    def __init__(self, filename):
        """
        初始数据集就是user和item 的评分矩阵  implicit
        """
        self.filename = filename
        self.data, self.shape = self.getData(filename)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()
        self.test_neg = self.getTestNeg(self.test, 99)

    def getData(self, filename):
        if self.filename == 'ml-1m' or self.filename == "ml-100k" or self.filename == 'ml-1m-test':
            print("Loading %s data set..." % (self.filename))
            data = []
            filePath = '../data/%s/ratings.dat' % (self.filename)
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("::")
                        user = int(lines[0])
                        movie = int(lines[1])
                        # score = float( lines[2])

                        score = 1 if float(lines[2]) > 0.5 else 0
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]

        elif self.filename == 'AGrocery' or self.filename == "AToys" or self.filename == 'ABeauty' or self.filename == "ABaby":
            print("Loading %s data set..." % (self.filename))
            data = []
            filePath = '../data/%s/%s_ratings.dat' % (
                self.filename, self.filename)
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split(",")
                        user = int(lines[0])
                        movie = int(lines[1])
                        # score = float(lines[2])
                        score = 1 if float(lines[2]) > 0.5 else 0
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u+1, i+1, len(data)))
            return data, [u+1, i+1]

        elif self.filename == 'AMusic' or self.filename == "pinterest-20":
            print("Loading %s data set..." % (self.filename))
            self.train = []
            filepath = '../data/%s/%s.train.rating' % (
                self.filename, self.filename)
            self.train_user_list = []
            self.train_item_list = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line:
                        lines = line.split('\t')
                        user = int(lines[0])
                        item = int(lines[1])
                        # score = float(lines[2])
                        score = 1 if float(lines[2]) > 0.5 else 0
                        self.train.append((user, item, score))
                        self.train_user_list.append(user)
                        self.train_item_list.append(item)

            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}  max user ID{} \n"
                  "\tItem Num: {}  max item ID{} \n"
                  "\tData Size: {}".format(
                      len(set(self.train_user_list)), max(set(self.train_user_list)), len(
                          set(self.train_item_list)), max(set(self.train_item_list)), len(self.train)))

            return self.train, [max(set(self.train_user_list))+1, max(set(self.train_item_list))+1]
        else:
            print("11Current data set is not support!")
            sys.exit()

    def getTrainTest(self):
        """
        test data
        """
        if self.filename == "ml-1m" or self.filename == "ml-100k" or self.filename == "ml-1m-test":
            data = self.data
            data = sorted(data, key=lambda x: (x[0], x[3]))
            train = []
            test = []
            for i in range(len(data)-1):
                user = data[i][0]-1
                item = data[i][1]-1
                rate = data[i][2]
                if data[i][0] != data[i+1][0]:
                    test.append((user, item, rate))
                else:
                    train.append((user, item, rate))

            test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
            return train, test

        elif self.filename == 'AGrocery' or self.filename == "AToys" or self.filename == 'ABeauty' or self.filename == "ABaby":
            data = self.data
            data = sorted(data, key=lambda x: (x[0], x[3]))
            train = []
            test = []
            for i in range(len(data)-1):
                user = data[i][0]
                item = data[i][1]
                rate = data[i][2]
                if data[i][0] != data[i+1][0]:
                    test.append((user, item, rate))
                else:
                    train.append((user, item, rate))

            test.append((data[-1][0], data[-1][1], data[-1][2]))
            return train, test
        elif self.filename == "AMusic" or self.filename == "pinterest-20":
            print("Loading %s data set..." % (self.filename))
            self.test = []
            filepath = '../data/%s/%s.test.rating' % (
                self.filename, self.filename)
            self.test_user_list = []
            self.test_item_list = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line:
                        lines = line.split('\t')
                        user = int(lines[0])
                        item = int(lines[1])
                        score = float(lines[2])
                        self.test.append((user, item, score))
                        self.test_user_list.append(user)
                        self.test_item_list.append(item)
            return self.train, self.test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def get_rateing_train_matrix(self):
        train_matrix = np.zeros(
            [self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, negNum):
        data = self.train
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        """
        返回的是一个list
        list[0] user的 二维数组
        list[1] item的 二维数据(第一项为test_positive item 后面的为negative item)
        """
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]

    def write_test_neg(self):
        """
        将测试负例写入文件
        """

        with open("../data/%s/%s.test.negative" % (self.filename, self.filename), 'w') as f:
            for i in range(len(self.test_neg[0])):
                user_list = self.test_neg[0][i]
                # print("userlist",user_list)
                item_list = self.test_neg[1][i]
                for j in range(len(user_list)):
                    if j == 0:
                        f.writelines("({},{})".format(
                            user_list[j], item_list[j]))
                    elif j < len(user_list)-1:
                        f.writelines("\t{}".format(item_list[j]))
                    else:
                        f.writelines("\t{}\n".format(item_list[j]))
        print("write test neg  OK")

        with open("../data/%s/%s.test.rating" % (self.filename, self.filename), 'w') as fi:
            for i in range(len(self.test)):
                fi.writelines("{}\t{}\t{}\n".format(self.test[i][0], self.test[i][1], self.test[i][2]))
        print("write test   OK")

        with open("../data/%s/%s.train.rating" % (self.filename, self.filename), 'w') as fi:
            for i in range(len(self.train)):
                fi.writelines("{}\t{}\t{}\n".format(
                    self.train[i][0], self.train[i][1], self.train[i][2]))
        print("write train   OK")


if __name__ == "__main__":
    # test_file = 'ml-1m'
    # dataSet = DataSet(test_file)
    # embedding = dataSet.get_rateing_train_matrix()
    # print("train", len(dataSet.train))
    # print("test", len(dataSet.test))
    # print(embedding.shape)

    test_file='AGrocery'
    dataSet=DataSet(test_file)
    embedding=dataSet.get_rateing_train_matrix()
    dataSet.write_test_neg()

    # print("train", len(dataSet.train))
    # print("test", len(dataSet.test))
    # print(embedding.shape)
    # print("train+test user num:",
    #       len(set(dataSet.train_user_list+dataSet.test_user_list)))
    # print("train+test item num:",
    #       len(set(dataSet.train_item_list+dataSet.test_item_list)))
    # print(1-46087/21643437)

    # test_file = 'pinterest-20'
    # dataSet = DataSet(test_file)
    # embedding = dataSet.get_rateing_train_matrix()
    # print("train", len(dataSet.train))
    # print("test", len(dataSet.test))
    # print(embedding.shape)
    # print("train+test user num:",
    #       len(set(dataSet.train_user_list+dataSet.test_user_list)))
    # print("train+test item num:",
    #       len(set(dataSet.train_item_list+dataSet.test_item_list)))
    # print(1-46087/21643437)
