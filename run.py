#! /usr/bin/env python

import sys
import getopt
import numpy
import os
import math
import pylab as pl
import sklearn
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def read(file):
    d = numpy.array([])
    if os.path.exists(file):
        d = numpy.genfromtxt(file,missing_values='?',delimiter=",")
        print type(d)
    else:
        print "Oh dear. File {} does not excist".format(file)
    return d

def normalise_values(d):
    average = [0 for i in range(len(d[0]))]
    count = [0 for i in range(len(d[0]))]
    dmax = [0 for i in range(len(d[0]))]
    dmin = [0 for i in range(len(d[0]))]
    for i in range(len(dmin)):
        dmin[i] = sys.float_info.max
    for i in range(len(d)):
        for j in range(len(d[i])):
            if math.isnan(d[i][j]) == 0:
                average[j] = average[j]+ d[i][j]
                count[j] = count[j]+1
                if d[i][j] > dmax[j]:
                    dmax[j] = d[i][j]
                if d[i][j] < dmin[j]:
                    dmin[j] = d[i][j]
                
    for i in range(len(average)):
        average[i] = average[i] / count[i]

    for i in range(len(d)):
        for j in range(len(d[i])-1):
            if math.isnan(d[i][j]):
                d[i][j] = average[j]
            if average[j]:
                d[i][j] = (d[i][j]-dmin[j])/(dmax[j]-dmin[j])

def get_class(d):
    dclass = [0 for i in range(len(d))]
    for i in range(len(d)):
       dclass[i] = d[i][len(d[i])-1]
    return dclass

def train(a,sizel,intercept):
    d = a.copy()    
    pes = Perceptron(n_jobs=4,n_iter=500,fit_intercept=intercept)
#    d = d.tolist()
    train = d[:len(d)/sizel]
    C = d[len(d)/sizel:]
    train_res = numpy.zeros(shape=(len(train)))#[0.0 for i in range(len(train))]
    C_res = numpy.zeros(shape=(len(C)))#[0.0 for i in range(len(C))]
#    C = [0.0 for i in range(len(C))]
    class_index = len(d[0])-1
    for i in range(len(train)):
        train_res[i] = (train[i][class_index] > 1)# and train[i][class_index] < 16)
        train[i][class_index] = 0        
        C_res[i] = (C[i][class_index]> 1)# and C[i][class_index] < 16)
        C[i][class_index] = 0
    
    pes.fit(train,train_res)
    output = pes.predict(C)
    (falsepr, truepr, thr) = roc_curve(C_res, output, 1)
    area = auc(falsepr, truepr)
    output = pes.score(C,C_res)
    return (output, area)
#    for i in range(len(output)):#            

def train_svm(data,sizel,intercept):
    d = data.copy()
    train = d[:len(d)/sizel]
    C = d[len(d)/sizel:]
    train_res = numpy.zeros(shape=(len(train)))
    C_res = numpy.zeros(shape=(len(C)))
    class_index = len(d[0])-1



    for i in range(len(train)):
        train_res[i] = (train[i][class_index] > 1)# and train[i][class_index] < 16)
        train[i][class_index] = 0        
        C_res[i] = (C[i][class_index]> 1)# and C[i][class_index] < 16)
        C[i][class_index] = 0
    
    C_range = 10.0 ** numpy.arange(-2, 9)
    gamma_range = 10.0 ** numpy.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedKFold(y=train_res, n_folds=3)
    grid = GridSearchCV(SVC(kernel=intercept,probability=True), param_grid=param_grid, cv=cv)
    svc = grid.fit(train, train_res,cv=5)

#    clf = svm.SVC(kernel=intercept,max_iter=-1,degree=3,shrinking=True,coef0=0.0,gamma=0.0,C=1.0, cache_size=2000)
 #   clf = clf.fit(train,train_res)
    output  = svc.predict(C)
    fpr, tpr, thr = roc_curve(C_res,output, 1)
    area = auc(fpr, tpr)
    output = grid.score(C,C_res)
    return (output, area)
    
def main(argv=None):
    print "hello"
    if argv is None:
        argv = sys.argv
    print "Opening file {}".format(argv[1])
    data = read(argv[1])
    if data.size == 0:
        print "Exit failure"
        return 2
    normalise_values(data)
    #dclass = get_class(data)
    correct_lin = []
    correct_rfb = []
    correct_lin_area = []
    correct_rfb_area = []
    
    correct_sig = []
    correct_sig_area = []
    correct_poly = []
    correct_poly_area = []
    #how many averages
    averages = 30
    #N/range, if 2 half of the data for training, 10 for one tenth of the data for training
    range_min = 2
    range_max = 11
    #changeme for PCA
    dopca = 1
    if dopca == 1:
        pca = PCA(n_components=len(data[0]),copy=True)
        pca.fit(data)
    #changeme to 1 for SVM 0 for perceptron
    dosvm = 0
    
    if dosvm == 0:
        for i in range(range_min,range_max):
            ta = 0
            taa = 0
            fa = 0
            faa = 0
            for j in range(averages):
                d = data.copy()
                numpy.random.shuffle(d)
                (toaverage, toarea) = train(d,i,True)
                (foaverage, foarea) = train(d,i,False)
                ta = ta + toaverage
                taa = taa + toarea
                
                fa = fa + foaverage
                faa = faa + foarea

            correct_sig_area.append([float(taa)/averages])
            correct_sig.append([float(ta)/averages])
            
            correct_poly_area.append([float(faa)/averages])
            correct_poly.append([float(fa)/averages])
        x2 = [float(1/float(i))*100 for i in range(range_min,range_max)]
        pl.subplot(210)
        pl.plot(x2,correct_sig)
        pl.plot(x2,correct_poly)
        pl.xlabel('Train size % of all data')
        pl.ylabel('Accuracy %')
        pca = ", PCA"
        pl.legend(['Centering', 'No Centering'])
        title = "Perceptron, mean accuracy"        
        if dopca ==0:
            pca = ""
        pl.title("{}{}".format(title,pca))
        pl.subplot(211)
        pl.xlabel('Train size % of all data')
        pl.ylabel('Area %')
        pca = ", PCA"
        pl.plot(x2,correct_sig_area)
        pl.plot(x2,correct_poly_area)
        pl.legend(['Centering', 'No Centering'])
        title = "Perceptron, area"        

        if dopca ==0:
            pca = ""
        pl.title("{}{}".format(title,pca))    
        pl.grid()
        pl.show()
                
    if dosvm == 1:
        for i in range(range_min,range_max):
            print i
            sla = 0
            slaa = 0
            sra = 0
            sraa = 0
            spa = 0
            spaa = 0
            ssa = 0
            ssaa = 0
            for j in range(averages):
                
                d = data.copy()
                numpy.random.shuffle(d)
                (loaverage, loarea) = train_svm(d.copy(),i,'linear')
                (roaverage, roarea) = train_svm(d.copy(),i,'rbf')
                (poaverage, poarea) = train_svm(d.copy(),i,'poly')
                (soaverage, soarea) = train_svm(d.copy(),i,'sigmoid')
                sla = sla + loaverage
                slaa = slaa + loarea
                
                sra = sra + roaverage
                sraa = sraa + roarea
                
                spa = spa + poaverage
                spaa = spaa + poarea

                ssa = ssa + soaverage
                ssaa = ssaa + soarea
            correct_lin.append([float(sla)/averages])
            correct_lin_area.append([float(slaa)/averages])

            correct_rfb.append([float(sra)/averages])
            correct_rfb_area.append([float(sraa)/averages])

            correct_poly.append([float(spa)/averages])
            correct_poly_area.append([float(spaa)/averages])

            correct_sig.append([float(ssa)/averages])
            correct_sig_area.append([float(ssaa)/averages])
        
        x2 = [float(1/float(i))*100 for i in range(range_min,range_max)]
        pl.subplot(210)
        pl.plot(x2,correct_lin)
        pl.plot(x2,correct_rfb)
        pl.plot(x2,correct_poly)
        pl.plot(x2,correct_sig)
        pl.xlabel('Train size % of all data')
        pl.ylabel('Accuracy %')
        pca = ", PCA"
        pl.legend(['linear', 'rbf','poly','sig'])
        title = "Support Vector Machine, mean accuracy"        
        if dopca ==0:
            pca = ""
        pl.title("{}{}".format(title,pca))
        pl.subplot(211)
        pl.plot(x2,correct_lin_area)
        pl.plot(x2,correct_rfb_area)
        pl.xlabel('Train size % of all data')
        pl.ylabel('Area %')
        pca = ", PCA"
        pl.plot(x2,correct_poly_area)
        pl.plot(x2,correct_sig_area)
        pl.legend(['linear', 'rbf','poly','sig'])
        title = "Support Vector Machine, area"        
        if dopca ==0:
            pca = ""
        pl.title("{}{}".format(title,pca))    
        pl.grid()
        pl.show()
    
if __name__ == "__main__":
    sys.exit(main())
