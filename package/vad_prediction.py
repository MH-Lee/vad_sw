import matplotlib.pyplot as plt
# from lightgbm import LGBMClassifier
from package.smoothing import smoothing
from package.clf import (logistic_regression,
                         neural_network,
                         gradient_boosting,
                         stacking_classifier,
                         voting_classifier)
from package.vad_filter import  feature_generation
from package.utils import (normalization,
                           infile_concat,
                           absolute)
from datetime import datetime
import pickle
import os
import pandas as pd


def plot_graph(origin, sm_pred, a=0, b=None, save_dir=None, img_name=None):
    plt.figure(figsize=(40,10))
    plt.plot(origin.loc[a:b], color='tab:red')
    plt.plot(sm_pred.loc[a:b], color='tab:blue')
    plt.legend(['Signal', 'Predict'])
    if save_dir is not None:
        plt.savefig(save_dir + '/img/{}.jpg'.format(img_name))
    else:
        plt.show()

def vad_preprocessing(df):
    n_person = df.shape[1]
    norm_df = normalization(df)
    absolute_df = absolute(norm_df)
    concat_df = infile_concat(absolute_df)
    return concat_df, n_person

def predict_result(test_x, model_name='lr'):
    model_name = "Stacking"
    model_name = model_name.lower()
    model_name
    if model_name == 'lr':
        sm_pred, origin = logistic_regression(test_x)
    elif model_name == 'mlp':
        sm_pred, origin = neural_network(test_x)
    elif model_name == 'grb':
        sm_pred, origin = gradient_boosting(test_x)
    elif model_name == 'voting':
        sm_pred, origin = voting_classifier(test_x)
    elif model_name == 'stacking':
        sm_pred, origin = stacking_classifier(test_x)
    else:
        sm_pred, origin = logistic_regression(test_x)
    return sm_pred, origin

def vad_predict(df, output_dir, img_name, n_person, model_name = 'LR'):
    df, n_person = vad_preprocessing(df)
    feature_df = feature_generation(df)
    test_x = feature_df.iloc[:,1:]
    model_name = model_name.lower()
    sm_pred, origin = predict_result(test_x, model_name)
    plot_graph(origin, sm_pred, save_dir=output_dir, img_name=img_name)
    pred_df = pd.concat([df['person'], sm_pred], axis=1)
    list_to_concat = []
    new_columns = []
    for p in range(1,  n_person+1):
        tmp_df = pred_df[pred_df['person'] == p].reset_index(drop=True)
        list_to_concat.append(tmp_df)
        new_columns += ['person_{}'.format(p), 'talking_{}'.format(p)]
    pred_df_concat = pd.concat(list_to_concat, axis=1)
    pred_df_concat.columns = new_columns
    return pred_df_concat
