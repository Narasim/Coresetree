import numpy as np
import pandas as pd
import math
import sys
import time
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn import datasets

clusters = {}
centroids = []
np_data = []

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(filename, m):
    #if data_type == 1:
    #	pass
    #filename = sys.argv[1] #dataset to calculate coreset of
    #output = sys.argv[2] #output file to print probability distribution values
    dataset_df = None
    if filename == "DataSets/bio_train.csv":
        dataset_df = pd.read_csv(filename,sep="\s+",header = None)
    elif filename == "DataSets/data_kddcup04/phy_train2.csv":
        dataset_df = pd.read_csv(filename,sep=",",header = None)
    elif filename == "DataSets/MiniBoone.csv":
        dataset_df = pd.read_csv(filename,sep=",")
    elif filename == "DataSets/HTRU2/HTRU_2.csv":
        dataset_df = pd.read_csv(filename,header = None)
    elif filename == "DataSets/shuttle/shuttle.xls":
        dataset_df = pd.read_excel(filename,sep="\s+",header = None)
    elif filename == "DataSets/default of credit card clients.csv":
        dataset_df = pd.read_csv(filename,sep=",",header = 0)
    elif filename == "DataSets/spambase/spambaseTrainTest.data":
        dataset_df = pd.read_csv(filename,sep=",",header = None)
    dim = dataset_df.shape[1]
    rows = dataset_df.shape[0]
    if filename == "DataSets/shuttle/shuttle.xls":
        data_df = dataset_df.iloc[:rows-10000, :dim] #full data with class values, removed more rows here to avoid maximum recursion limit.
    else:
        data_df = dataset_df.iloc[:rows-1000, :dim] #full data with class values
    #class_df = dataset_df.iloc[:rows-1, 0:1]
    #print(class_df)
    #print(data_df.head())
    rows = data_df.iloc[:] #all the rows in selected dataset
    #print(data_df)
    data_size = len(rows) #calculating #no. of entries in data(no. of rows)
    if filename == "DataSets/bio_train.csv":
        data = np.array(data_df.iloc[:,3:dim]) # choosing dataset without class
    elif filename == "DataSets/data_kddcup04/phy_train2.csv":
        data = np.array(data_df.iloc[:,2:dim]) # choosing dataset without class
    elif filename == "DataSets/MiniBoone.csv":
        data = np.array(data_df.iloc[:,:dim-1])
    elif filename == "DataSets/HTRU2/HTRU_2.csv":
        data = np.array(data_df.iloc[:,:dim-1]) # choosing dataset without class column
    elif filename == "DataSets/shuttle/shuttle.xls":
        data = np.array(data_df.iloc[:,:dim-1]) # choosing dataset without class column
    elif filename == "DataSets/default of credit card clients.csv":
        data = np.array(data_df.iloc[:,1:dim-1]) # choosing dataset without class column
    elif filename == "DataSets/spambase/spambaseTrainTest.data":
        data = np.array(data_df.iloc[:,:dim-1]) # choosing dataset without class column
    #print(data)
    data_mean = np.mean(data, axis = 0)
    #print(data_mean)
    distance = 0
    for point in data:
        distance += np.sum(np.square(point-data_mean))
    #print(distance)
    prob_dist = []
    #calculating proposal  distribution for each row
    for point in data:
        value=((0.5*(1/data_size))+0.5*((np.sum(np.square(point-data_mean)))/distance))
        prob_dist.append(value)
    df = pd.DataFrame(prob_dist)
    #print(prob_dist)
    data_df['Prob_dist'] = df
    #print(data_df)
    #writing ProbDist to file
    #dataset.to_csv("//home//oseen//Documents//Mtech_project//code2_KDDdata2004//light_bio_train1.csv",index=False)
    #adding weight value to dataset
    weight_value = []
    for i in range(data_size):
        #print("i is: ",i," m is ",m,"prob_dist is ",prob_dist[0]," type is ",prob_dist[0].dtype)
        weight = 1/(m*prob_dist[i])
        weight_value.append(weight)
    df = pd.DataFrame(weight_value)
    data_df['weight_value'] = df
    #class_df['weight_value'] = df
    #print(class_df)
    #dataset.to_csv("//home//oseen//Documents//Mtech_project//code2_KDDdata2004//light_bio_train1.csv",index=False)
    #sorting result
    #sorted_data = data_df.sort_values('weight_value',ascending='False')
    #sorted_class = class_df.sort_values('weight_value',ascending='False')
    #print(sorted_data.iloc[:m,:dim])
    #print(sorted_class.iloc[:m,:dim-1])
    #print(sorted_class.iloc[13])
    sampled_points = data_df.sample(n=m, frac=None, replace=False, weights='weight_value', random_state=None, axis=None)
    return sampled_points.iloc[:,:dim]
    #return sorted_data.iloc[:m,:dim]



if __name__ == "__main__":
    #calling generate_data() for data to be generated/read.
    if len(sys.argv) != 3:
        print("use python3 programname.py <dataset_name> <size of coreset 'm'>to run.")
        exit()
    filename = sys.argv[1] #dataset to calculate coreset of
    m = int(sys.argv[2])
    start_time = time.time()
    data_with_class = generate_data(filename, m) #dataset with class variables
    dim = data_with_class.shape[1]
    #print(data_with_class)
    if filename == "DataSets/bio_train.csv":
        data = data_with_class.iloc[:,3:dim] #data without class variable
        df = pd.read_csv(filename,sep="\s+")
    elif filename == "DataSets/data_kddcup04/phy_train2.csv":
        data = data_with_class.iloc[:,2:dim] #data without class variable
        df = pd.read_csv(filename,sep=",")
    elif filename == "DataSets/MiniBoone.csv":
        data = data_with_class.iloc[:,:dim-1] #data without class variable
        df = pd.read_csv(filename,sep=",")
    elif filename == "DataSets/HTRU2/HTRU_2.csv":
        data = data_with_class.iloc[:,:dim-1] #data without class variable
        df = pd.read_csv(filename,sep=",")
    elif filename == "DataSets/shuttle/shuttle.xls":
        data = data_with_class.iloc[:,:dim-1] #data without class variable
        df = pd.read_excel(filename,sep="\s+")
    elif filename == "DataSets/default of credit card clients.csv":
        data = data_with_class.iloc[:,1:dim-1] #data without class variable
        df = pd.read_csv(filename,sep=",")
    elif filename == "DataSets/spambase/spambaseTrainTest.data":
        data = data_with_class.iloc[:,:dim-1] #data without class variable
        df = pd.read_csv(filename,sep=",")
    #df = pd.read_csv(filename,sep="\s+")
    dim = df.shape[1]
    rows = df.shape[0]
    leafsize = 50
    tree = spatial.KDTree(data, leafsize)
    #time in building index(offlinePhase)
    print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
    rightGuessCount = 0
    maxTime = -1000;
    minTime = 1000;
    totalTime = 0;
    for i in range(1,1000):
        query_point_with_class = df.iloc[rows-i:rows-(i-1), :dim] #query_point dataframe with class
        #building tree based on given points_list and leaf_size
        if filename == "DataSets/bio_train.csv":
            query_point = np.array(query_point_with_class.iloc[:,3:dim]) # using query_point without class variable
        elif filename == "DataSets/data_kddcup04/phy_train2.csv":
            query_point = np.array(query_point_with_class.iloc[:,2:dim]) # using query_point without class variable
        elif filename == "DataSets/MiniBoone.csv":
            query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
        elif filename == "DataSets/HTRU2/HTRU_2.csv":
            query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
        elif filename == "DataSets/shuttle/shuttle.xls":
            query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
        elif filename == "DataSets/default of credit card clients.csv":
            query_point = np.array(query_point_with_class.iloc[:,1:dim-1]) # using query_point without class variable
        elif filename == "DataSets/spambase/spambaseTrainTest.data":
            query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
        #print("Data dimensions: "+str(data.shape))
        #starting time count
        start_time = time.time()
        k = 50
        dist,indices = (tree.query(query_point, k))
        #printing nearest neighbors
        #list of indices is indices[0]
        nnClassList = []
        #print("Nearest Points to the query are: ")
        for index in indices[0]:
            #change to appropriate class column based on the dataset
            if filename == "DataSets/bio_train.csv":
                nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][2])]) #colm 2 is class here.
            elif filename == "DataSets/data_kddcup04/phy_train2.csv":
                nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][1])]) #col 1 represents class here.
            else:
                nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][dim-1])]) #last colmn represents class here.
        #print(nnClassList)
        uniqw, inverse = np.unique(nnClassList, return_inverse=True)
        #print("unique inverse ",uniqw, inverse)
        arr = np.bincount(inverse)
        indexOfMaxOccur = np.where(arr == max(np.bincount(inverse)))
        newClass = uniqw[indexOfMaxOccur[0][0]]  #indexOfMaxOccur is a list of one numpyArray with newClass as its first and only element. [0] accesses, numpy array and another [0] access actual index.
        #change to appropriate class column based on the dataset
        if filename == "DataSets/bio_train.csv":
            aClass = np.array(query_point_with_class)[0][2] #col 2 represents class here.
        elif filename == "DataSets/data_kddcup04/phy_train2.csv":
            aClass = np.array(query_point_with_class)[0][1] #col 1 represents class here.
        else:
            aClass = np.array(query_point_with_class)[0][dim-1] # last col of data represents class here.
        #print(aClass)
        if aClass == newClass:
            rightGuessCount += 1
            #print("right ", rightGuessCount, "Times")
        #else:
            #print("WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG")
        totalTime += (time.time() - start_time)
        if maxTime < (time.time() - start_time):
            maxTime = (time.time() - start_time)
        if minTime > (time.time() - start_time):
            minTime = (time.time() - start_time)
        #print("--- %s seconds ---" % ((time.time() - start_time)))
    print("RightGuesses: ", rightGuessCount, " MaxTime: ",maxTime, " MinTime: ",minTime, " AvgTime: ",totalTime/1000)
