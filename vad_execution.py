import pandas as pd
import numpy as np
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


fpath = './data/3p_1(test).csv'
# fpath = './data/2p_1(test).csv'
# fpath = './data/2p_6(test).csv'
# fpath = './data/2p_7(test_4).csv'
# fpath = './data/2p_2.csv'
output_dir = './results/'
model_name = 'Stacking'
tt_term = 300

def final_excecution(fpath, output_dir, model_name, tt_term):
    fname = os.path.basename(fpath).split('.')[0]
    file_extension = os.path.basename(fpath).split('.')[1]

    ### Try two type of
    if file_extension == 'csv':
        df = pd.read_csv(fpath)
    else:
        df = pd.read_excel(fpath)

    n_person = df.shape[1]
    df_columns = ['t{}_AMP'.format(i) for i in range(1, n_person+1)]
    df_new_columns = ['x{}'.format(i) for i in range(1, n_person+1)]
    df = df[df_columns]
    df.columns = df_new_columns
    ### predict vad
    pred_df_concat = vad_predict(df, output_dir, \
                                n_person=n_person, \
                                img_name=fname,
                                model_name=model_name)

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

    ### Turn taking matrix
    turn_taking_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='turn_taking')
    short_res_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='short_res')
    tt_and_short = turn_taking_df + short_res_df
