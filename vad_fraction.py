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
                                      make_turn_taking_df,
                                      combi_suspect,
                                      silence_breaker,
                                      silence_table,
                                      mirroring,
                                      mirroring_matrix)
from itertools import permutations, chain
from collections import Counter
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
s_term = 300
m_term = 300
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



combi = combi_suspect(only_talk, n_person=n_person)
sil_breaker = silence_breaker(only_talk, combi, n_person=n_person, howlong=s_term)
s_breaker_df = silence_table(sil_breaker, n_person=n_person)
s_breaker_df

mirroring_matrix(dialog_len_list, n_person, period=m_term)
from datetime import datetime

today = datetime.today()

# def breaker(df, combi, n_person, howlong = 300):
#     """
#     - combi : the result value of combi_suspect function
#     - howlong : the time period during silence
#
#     - returns person indices who breaks a silence
#     - person index : person1 --> 0, person2 --> 1, person3 -->2
#
#     """
#     comb_silence_values = combi.values
#     who_talk = []
#     #value_who_talk = []
#     acc = []
#     select_columns = ['talk{}'.format(i) for i in range(1, n_person+1)]
#     for i, value in enumerate(comb_silence_values):
#         if(comb_silence_values[i,0] == 0):
#             pass
#         elif(comb_silence_values[i,0] - comb_silence_values[i-1,0] >= 2): #침묵이 깨졌을 떄
#             if(comb_silence_values[i-1,1] <= -howlong): #침묵이 300 이상 진행 되었을 떄,
#                 break_idx = comb_silence_values[i-1,0] + 1 # 침묵을 깬 사람이 발생한 row의 index
#                 til_break_df = df[:break_idx + 1].loc[:,select_columns] # 첫 row 부터 침묵을 깬 row 까지 데이터프레임 슬라이스
#                 # 만일 til_break_df 에서 row의 원소들의 총합이 -3이 아닌 경우가 단 하나만 있다면, 해당 break_idx는 밑에서 어팬드 하지 않고 pass
#                 til_break_df['sum'] = til_break_df.sum( axis = 1)
#
#
#                 # 가장 처음 발생한 breaker는 silence breaker가 아니라, 단순히 처음 대화를 시작한 경우일 뿐이므로 제외
#                 if len(np.where(til_break_df['sum'] != -3)[0]) >= 2:
#                     acc.append(break_idx) # 침묵이 300 이상 진행 되었을때, 그것이 깨진 행을 append한다.
#
#     for ii, vv in enumerate(acc):
#         ## ex. [-1, 1, 1] 중에서 1,1에 대응되는 person을 구해라
#         # t = [df['talk1'][vv],df['talk2'][vv], df['talk3'][vv]]
#         t = list()
#         for c in select_columns:
#             t.append(df[c][vv])
#         tt = [i for i in range(len(t)) if t[i] == 1.0]
#         who_talk.append(tt)
#     return who_talk
#
#
# def table(whois, n_person):
#     """
#     - input : 아래 두 함수의 결과 --> breaker(combi_suspect(df), 400) 를 input으로 함
#     - output: person 별로 침묵을 깬 횟수 카운팅
#
#     creating a final table for silence breaker
#     """
#     z = list(chain(*whois))
#     if z == []:
#         return pd.DataFrame()
#     df = pd.DataFrame.from_dict(Counter(z), orient='index').reset_index()
#     df.columns = ['person', 'interruption']
#     df.sort_values(by='person', ascending=True, inplace = True)
#     df_final = df.copy()
#     for p in range(n_person):
#         df_final.loc[:, 'person'].replace(p, 'person{}'.format(p+1), inplace = True)
#     return df_final.reset_index(drop=True)
#
# def combi_suspect(df):
#     """
#     input: df = pd.read_csv('3_true_for_check.csv') 와 같이 원본 데이터 프레임을 넣는다.
#     output:
#
#         - creating a two-columned dataframe
#         - 1st column : a row corresponding to the index of p1&p2&p3, being silent
#             - each row of the 1st column value is -1, meaning silence
#             - 1st column consists of original index of the Scenario file
#         - 2nd column :
#             - continuously adding -1 as the line goes by
#             - stop accumulating -1 when the index jumps more than 2 steps
#             - it means there is a silence breaker
#     """
#     silent_times = []
#     df_sum_row = df.sum(axis=1)
#     for i in range(len(df)):
#         if (df_sum_row[i]) == -1* int(n_person):  ### 총합이 -3일때 침묵인 상황
#             silent_times.append(i)        ### -1-1-1 침묵인 row 인덱스를 어팬드
#
#     pd_silence = pd.DataFrame(silent_times.copy())
#     pd_silence.columns = ['idx']
#     pd_silence
#
#     # silence 옆에 추가할, -1 accumulation 한 column 생성
#     acc = []
#     total = 0
#     silence = 0
#
#     for idx, value in enumerate(pd_silence.iloc[:,0]):
#         total += silence
#         if(value == 0): #첫 행의 경우 -1-1-1 침묵일 경우임에 확실.
#             silence = -1
#             acc.append(silence)
#         elif(pd_silence.iloc[idx, 0] - pd_silence.iloc[idx-1, 0] >= 2):
#             silence = -1
#             acc.append(silence)
#         elif(pd_silence.iloc[idx, 0] - pd_silence.iloc[idx-1, 0] < 2):
#             silence += -1
#             acc.append(silence)
#
#     acc_silence = pd.DataFrame(acc.copy())
#     acc_silence.columns = ['acc_idx']
#     comb_silence = pd.concat([pd_silence, acc_silence], axis = 1)
#     return comb_silence
#
# def mirroring(host1, guest1, period = 350):
#     """
#     - 아래 mirroring_matrix 의 내장 함수로 쓰임
#     - input : detect_talk_break_length 의 결과값 중 두 개만 pair로 들어감 : 첫 번째로 들어가는 결과값이 host1, 그 다음에 들어가는 결과값이 guest1
#     - output : host1의 음성구간 동안 guest1 의 미러링 횟수를 카운팅
#
#     """
#     mirror2 = 0
#     for i in range(len(get_start_end_times(host1, guest1)[0])):# 호스트 기준으로만 미러링을 센다
#         for k in range(len(get_start_end_times(host1, guest1)[1])):
#             # host의 시작점 보다 guest의 시작점 index가 커야함 & host 의 끝점보다 guest의 시작점 index가 작아야함
#             if (get_start_end_times(host1, guest1)[0][i][0] < get_start_end_times(host1, guest1)[1][k][0]) and (get_start_end_times(host1, guest1)[0][i][1] > get_start_end_times(host1, guest1)[1][k][0]):
#                 # host의 시작점보다 guset의 끝점 index가 커야함  & host의 끝점보다 guest의 끝점 index가 작아야함
#                 if (get_start_end_times(host1, guest1)[0][i][0] < get_start_end_times(host1, guest1)[1][k][1]) and (get_start_end_times(host1, guest1)[0][i][1] > get_start_end_times(host1, guest1)[1][k][1]):
#                     if get_start_end_times(host1, guest1)[1][k][1] - get_start_end_times(host1, guest1)[1][k][0] <= period:
#                         mirror2 += 1
#     return mirror2
#
# # 포루프 돌려야 할 대상 : dialog_len1, dialog_len2, dialog_len3
# def mirroring_matrix(dialog_len_list, n_person, period = 350):
#     """
#     input : detect_talk_break_length 결과를 talk1 talk2 talk3 순서 대로 넣어준다.
#     output : mirroring 횟수를 나타내는 데이터프레임 출력
#
#     """
#     new_columns = ['to_p{}'.format(i) for i in range(1, n_person+1)]
#     np_data = np.zeros((n_person, n_person))
#     for idx, pm in enumerate(permutations(range(0, n_person), 2)):
#         # print(pm)
#         # print(pm[0]+1, "->", pm[1]+1)
#         np_data[pm] = mirroring(dialog_len_list[pm[0]],\
#                                 dialog_len_list[pm[1]],\
#                                 period)
#     mirroring_df =pd.DataFrame(np_data, columns=new_columns, index=new_columns)
#     return mirroring_df


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
