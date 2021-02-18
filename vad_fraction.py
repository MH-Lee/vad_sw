import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# # from lightgbm import LGBMClassifier
# from package.smoothing import smoothing
# from package.clf import (logistic_regression,
#                          neural_network,
#                          gradient_boosting,
#                          stacking_classifier,
#                          voting_classifier,
#                          lightGBM)
# from package.vad_filter import  feature_generation
# from package.utils import (normalization,
#                            infile_concat,
#                            absolute)
from package.vad_prediction import vad_predict
from package.dialog_detection import (post_smth,
                                      detect_talk_break_length,
                                      get_start_end_times,
                                      count_turn_takes,
                                      make_turn_taking_df)
from itertools import permutations
import warnings
import os
warnings.filterwarnings('ignore')




# df = pd.read_csv('./data/2p_1(test).csv')
# df = df[['t1_AMP','t2_AMP']]
# df.columns = ['x1', 'x2']

fpath = './data/3p_1(test).csv'
# fpath = './data/2p_1(test).csv'
# fpath = './data/2p_6(test).csv'
# fpath = './data/2p_7(test_4).csv'
# fpath = './data/2p_2.csv'
output_dir = './results/'
model_name = 'Stacking'
tt_term = 300
fname = os.path.basename(fpath).split('.')[0]
file_extension = os.path.basename(fpath).split('.')[1]

if file_extension == 'csv':
    df = pd.read_csv(fpath)
else:
    df = pd.read_excel(fpath)


n_person = df.shape[1]
df_columns = ['t{}_AMP'.format(i) for i in range(1, n_person+1)]
df_new_columns = ['x{}'.format(i) for i in range(1, n_person+1)]
df = df[df_columns]
df.columns = df_new_columns

# if n_person == 3:
#     df = df[['t1_AMP','t2_AMP', 't3_AMP']]
#     df.columns = ['x1', 'x2', 'x3']
# elif n_person == 2:
#     df = df[['t1_AMP','t2_AMP']]
#     df.columns = ['x1', 'x2']
# else:
#     raise("Consider only 2 or 3 person conversation")

pred_df_concat = vad_predict(df, output_dir, n_person=n_person, img_name=fname, model_name='Stacking')

#### Dialog ditection part
# pair-wise
# replace 0s to -1s for future break count algorithm
pred_df_concat

dd_df = pred_df_concat.copy()
for p in range(1, n_person+1):
    dd_df['talk{}'.format(p)] = dd_df['talking_{}'.format(p)].replace(0,-1)

### remove outlier (-1,-1,-1,-1,-1,-1)
new_df = post_smth(dd_df, n_person=n_person)
only_talk_list = list()
dialog_len_list = list()
for p in range(1, n_person + 1):
    only_talk_list.append(new_df['talk{}'.format(p)].reset_index(drop=True))
    dl_temp = detect_talk_break_length(new_df['talk{}'.format(p)])
    dialog_len_list.append(dl_temp)
only_talk = pd.concat(only_talk_list, axis=1)

# detail_result = count_tt_detail(only_talk, dialog_len_list, n_person, tt_term=10)
# detail_result
tt_term = 300
turn_taking_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='turn_taking')
short_res_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='short_res')
tt_and_short = turn_taking_df + short_res_df

turn_taking_df
short_res_df
tt_and_short


def table(whois, n_person):
    """
    - input : 아래 두 함수의 결과 --> breaker(combi_suspect(df), 400) 를 input으로 함
    - output: person 별로 침묵을 깬 횟수 카운팅

    creating a final table for silence breaker
    """
    z = list(itertools.chain(*whois))
    df = pd.DataFrame.from_dict(Counter(z), orient='index').reset_index()
    df.columns = ['person', 'interruption']
    df.sort_values(by='person', ascending=True, inplace = True)
    df_final = df.copy()
    for p in range(n_person):
        df_final.loc[:, 'person'].replace(0, 'person{}'.format(p+1), inplace = True)
    return df_final

def combi_suspect(df):
    """
    input: df = pd.read_csv('3_true_for_check.csv') 와 같이 원본 데이터 프레임을 넣는다.
    output:

        - creating a two-columned dataframe
        - 1st column : a row corresponding to the index of p1&p2&p3, being silent
            - each row of the 1st column value is -1, meaning silence
            - 1st column consists of original index of the Scenario file
        - 2nd column :
            - continuously adding -1 as the line goes by
            - stop accumulating -1 when the index jumps more than 2 steps
            - it means there is a silence breaker
    """
    silent_times = []
    for i in range(len(df)):
        if (df.sum(axis=1)[i]) == -3:  ### 총합이 -3일때 침묵인 상황
            silent_times.append(i)        ### -1-1-1 침묵인 row 인덱스를 어팬드

    pd_silence = pd.DataFrame(silent_times.copy())
    pd_silence.columns = ['idx']
    pd_silence

    # silence 옆에 추가할, -1 accumulation 한 column 생성
    acc = []
    total = 0
    silence = 0

    for idx, value in enumerate(pd_silence.iloc[:,0]):
        total += silence
        if(value == 0): #첫 행의 경우 -1-1-1 침묵일 경우임에 확실.
            silence = -1
            acc.append(silence)
        elif(pd_silence.iloc[idx, 0] - pd_silence.iloc[idx-1, 0] >= 2):
            silence = -1
            acc.append(silence)
        elif(pd_silence.iloc[idx, 0] - pd_silence.iloc[idx-1, 0] < 2):
            silence += -1
            acc.append(silence)

    acc_silence = pd.DataFrame(acc.copy())
    acc_silence.columns = ['acc_idx']
    comb_silence = pd.concat([pd_silence, acc_silence], axis = 1)
    return comb_silence

# np_data
# detail_result[0][0] # 1->2
# detail_result[1][0] # 1->3
# detail_result[2][0] # 2->1
# detail_result[3][0] # 2->3
# detail_result[4][0] # 3->1
# detail_result[5][0] # 3->2


# df, n_person = vad_preprocessing(df)
# feature_df = feature_generation(df)
# test_x = feature_df.iloc[:,1:]
# model_name = 'Stacking'
# sm_pred, origin = predict_result(test_x, model_name)
# plot_graph(origin, sm_pred)
#
# pred_df = pd.concat([df['person'], sm_pred], axis=1)
# list_to_concat = []
# new_columns = []
# for p in range(1,  n_person+1):
#     tmp_df = pred_df[pred_df['person'] == p].reset_index(drop=True)
#     list_to_concat.append(tmp_df)
#     new_columns += ['person_{}'.format(p), 'talking_{}'.format(p)]
# pred_df_concat = pd.concat(list_to_concat, axis=1)
# pred_df_concat.columns = new_columns
# pred_df_concat

# if model_name == 'lr':
#     sm_pred, origin = logistic_regression(test_x)
# elif model_name == 'mlp':
#     sm_pred, origin = neural_network(test_x)
# elif model_name == 'grb':
#     sm_pred, origin = gradient_boosting(test_x)
# elif model_name == 'lightgbm':
#     sm_pred, origin = lightGBM(test_x)
# elif model_name == 'voting':
#     sm_pred, origin = voting_classifier(test_x)
# elif model_name == 'stacking':
#     sm_pred, origin = stacking_classifier(test_x)
# else:
#     sm_pred, origin = logistic_regression(test_x)
