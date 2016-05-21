# -*- coding: utf-8 -*-

import numpy as np
import hashlib
from pylab import *
from sklearn import metrics
from collections import defaultdict

def fpToNP(fp,fpsz):
    nfp = np.zeros((fpsz,),np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nfp[int(hashlib.md5(str(idx).encode('utf-8')).hexdigest()[:8], 16) % fpsz] += v
    return nfp


def evaluateModel(model, testFPs, testReactionTypes, rTypes, names_rTypes):
    
    preds = model.predict(testFPs)
    newPreds=[rTypes[x] for x in preds]
    newTestActs=[rTypes[x] for x in testReactionTypes]
    cmat=metrics.confusion_matrix(newTestActs,newPreds)
    cmat=metrics.confusion_matrix(testReactionTypes,preds)
    colCounts = sum(cmat,0)
    rowCounts = sum(cmat,1)

    print ('%2s %7s %7s %7s     %s'%("ID","recall","prec","F-score ","reaction class"))
    sum_recall=0
    sum_prec=0
    for i,klass in enumerate(rTypes):
        recall = 0
        if rowCounts[i] > 0:
            recall = float(cmat[i,i])/rowCounts[i]
        sum_recall += recall
        prec = 0
        if colCounts[i] > 0:
            prec = float(cmat[i,i])/colCounts[i]
        sum_prec += prec
        f_score = 0
        if (recall + prec) > 0:
            f_score = 2 * (recall * prec) / (recall + prec)   
        print ('%2d % .4f % .4f % .4f % 9s %s'%(i,recall,prec,f_score,klass,names_rTypes[klass]))
    
    mean_recall = sum_recall/len(rTypes)
    mean_prec = sum_prec/len(rTypes)
    if (mean_recall+mean_prec) > 0:
        mean_fscore = 2*(mean_recall*mean_prec)/(mean_recall+mean_prec)
    print ("Mean:% 3.2f % 7.2f % 7.2f"%(mean_recall,mean_prec,mean_fscore))
    
    return cmat

