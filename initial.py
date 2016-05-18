from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import cPickle,gzip
from collections import defaultdict
import createFingerprintsReaction
import random
import time

starttime = time.time() # start time


###########################################################
# Combine AP3 fingerprint with agent feature and Morgan2 FPs
###########################################################

infile = gzip.open('training_test_set_patent_data.pkl.gz', 'rb')
pklfile = gzip.open('transformationFPs_MG2_agentFPs_test_set_patent_data.pkl.gz','wb+')

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    try:
        rxn = AllChem.ReactionFromSmarts(smi,useSmiles=True)
        fp_AP3 = createFingerprintsReaction.create_transformation_FP(rxn,AllChem.FingerprintType.AtomPairFP)
        fp_MG2_agents = createFingerprintsReaction.create_agent_morgan2_FP(rxn)
        if fp_MG2_agents is None:
            fp_MG2_agents = DataStructs.UIntSparseIntVect(4096)
        fp_featureAgent = createFingerprintsReaction.create_agent_feature_FP(rxn)
    except:
        print "Cannot build fingerprint/reaction of: %s\n"%smi
        continue;
    cPickle.dump((lbl,klass,fp_AP3,fp_featureAgent,fp_MG2_agents),pklfile,2)
    if not lineNo%5000:
        print "[%6.1fs] creating transformation FP - %d"%(time.time()-starttime, lineNo)


###########################################################
# Load the AP3 fingerprint, agent feature and MG2 fingerprints
###########################################################

from sklearn.linear_model import LogisticRegression
import utilsFunctions

infile = gzip.open("transformationFPs_MG2_agentFPs_test_set_patent_data.pkl.gz", 'rb')

lineNo=0
fps=[]
idx=0
while 1:
    lineNo+=1
    try:
        lbl,cls,fp_AP3,fp_agentFeature,fp_agentMG2 = cPickle.load(infile)        
    except EOFError:
        break
    fps.append([idx,lbl,cls,fp_AP3,fp_agentFeature,fp_agentMG2])
    idx+=1
    if not lineNo%10000:
        print "[%6.1fs] loading pickle file - %d"%(time.time()-starttime, lineNo)


###########################################################
# Split the FPs in training (20 %) and test data (80 %)
###########################################################

import numpy as np

random.seed(0xd00f)
indices=range(len(fps))
random.shuffle(indices)

nActive=200
fpsz=256
trainFps_AP3_agentMG2=[]
testFps_AP3_agentMG2=[]
trainActs=[]
testActs=[]

reaction_types = cPickle.load(file("reactionTypes_training_test_set_patent_data.pkl"))
names_rTypes = cPickle.load(file("names_rTypes_classes_superclasses_training_test_set_patent_data.pkl"))

rtypes=sorted(list(reaction_types))
for i,klass in enumerate(rtypes):
    actIds = [x for x in indices if fps[x][2]==klass]
    for x in actIds[:nActive]:
        # np1_feature = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        # np2_feature = np.asarray(fps[x][4], dtype=float)
        # trainFps_AP3_agentFeature += [np.concatenate([np1_feature, np2_feature])]
        np1_morgan = utilsFunctions.fpToNP(fps[x][3],fpsz)
        # trainFps_AP3 += [np1_morgan]
        np2_morgan = utilsFunctions.fpToNP(fps[x][5],fpsz)
        trainFps_AP3_agentMG2 += [np.concatenate([np1_morgan, np2_morgan])]
    trainActs += [i]*nActive
    nTest=len(actIds)-nActive
    for x in actIds[nActive:]:
        # np1_feature = utilsFunctions.fpToNPfloat(fps[x][3],fpsz)
        # np2_feature = np.asarray(fps[x][4], dtype=float)
        # testFps_AP3_agentFeature += [np.concatenate([np1_feature, np2_feature])]
        np1_morgan = utilsFunctions.fpToNP(fps[x][3],fpsz)
        # testFps_AP3 += [np1_morgan]
        np2_morgan = utilsFunctions.fpToNP(fps[x][5],fpsz)
        testFps_AP3_agentMG2 += [np.concatenate([np1_morgan, np2_morgan])]
    testActs += [i]*nTest
    
print "[%6.1fs] splited FP collection to training and test set"%(time.time()-starttime)


###########################################################
# Train LR Model
###########################################################

lr_cls_AP3_MG2 = LogisticRegression()
result_lr_fp_AP3_MG2 = lr_cls_AP3_MG2.fit(trainFps_AP3_agentMG2,trainActs)
print "[%6.1fs] LR model training finished"%(time.time()-starttime)


###########################################################
# Evaluate Model
###########################################################

cmat_fp_AP3_MG2 = utilsFunctions.evaluateModel(result_lr_fp_AP3_MG2, testFps_AP3_agentMG2, testActs, rtypes, names_rTypes)
print "[%6.1fs] evaluation finished!"%(time.time()-starttime)
