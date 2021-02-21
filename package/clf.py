import pandas as pd
import pickle
import os
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             recall_score,
                             confusion_matrix)
from .smoothing import smoothing

cur_dir = os.getcwd()

def logistic_regression(test_x) :
    filename = cur_dir + '/models/lr_clf_vad.pkl'
    lr =  pickle.load(open(filename, 'rb'))
    lr_pred = lr.predict_proba(test_x) # test 셋을 학습한 모델에 넣어서 확률값을 받는다.
    lr_sm_pred = smoothing(lr_pred[:,1]).reset_index(drop=True)  # 확률값들을 스무딩한다.
    lr_origin = pd.DataFrame(lr_pred[:,1]).reset_index(drop=True)
    return lr_sm_pred, lr_origin

def neural_network(test_x):
    filename = cur_dir + '/models/mlp_clf_vad.pkl'
    nn = pickle.load(open(filename, 'rb'))
    nn_pred = nn.predict_proba(test_x)
    nn_sm_pred = smoothing(nn_pred[:,1])  # 확률값들을 스무딩한다.
    nn_origin = pd.DataFrame(nn_pred[:,1]).reset_index(drop=True)
    return nn_sm_pred, nn_origin

def gradient_boosting(test_x):
    filename = cur_dir + '/models/grb_clf_vad.pkl'
    grb = pickle.load(open(filename, 'rb'))
    grb_pred = grb.predict_proba(test_x)
    grb_sm_pred = smoothing(grb_pred[:,1])  # 확률값들을 스무딩한다.
    grb_origin = pd.DataFrame(grb_pred[:,1]).reset_index(drop=True)
    return grb_sm_pred, grb_origin

def lightGBM(test_x) :
    filename = cur_dir + '/models/lgb_clf_vad.pkl'
    lgb_clf = pickle.load(open(filename, 'rb'))
    lgb_pred = lgb_clf.predict_proba(test_x)
    lgb_sm_pred = smoothing(lgb_pred[:,1])
    lgb_origin = pd.DataFrame(lgb_pred[:,1]).reset_index(drop=True)
    return lgb_sm_pred, lgb_origin


def voting_classifier(test_x):
    filename = cur_dir + '/models/voting_clf_vad.pkl'
    voting_clf = pickle.load(open(filename, 'rb'))
    voting_pred = voting_clf.predict_proba(test_x)
    voting_sm_pred = smoothing(voting_pred[:,1])
    voting_origin = pd.DataFrame(voting_pred[:,1]).reset_index(drop=True)
    return voting_sm_pred, voting_origin


def stacking_classifier(true_x):
    filename = cur_dir + '/models/stacking_clf_vad.pkl'
    stacking_clf = pickle.load(open(filename, 'rb'))
    stacking_pred = stacking_clf.predict_proba(true_x)
    stacking_sm_pred = smoothing(stacking_pred[:,1])
    stacking_origin = pd.DataFrame(stacking_pred[:,1]).reset_index(drop=True)
    return stacking_sm_pred, stacking_origin
