import pandas as pd
import numpy as np
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
                                      mirroring_matrix,
                                      turn_taking_matrix,
                                      )
import matplotlib.pyplot as plt
from itertools import permutations
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')


def final_excecution(fpath, output_dir, model_name, tt_term, s_term, m_term):
    fname = os.path.basename(fpath).split('.')[0]
    file_extension = os.path.basename(fpath).split('.')[1]
    today = datetime.strftime(datetime.today(), format="%Y%m%d")
    subdir_name = '/{}_{}({}_{}_{})'.format(today, model_name, tt_term, s_term, m_term)
    if not os.path.exists(output_dir +  subdir_name):
        os.makedirs(output_dir + subdir_name)
    if not os.path.exists(output_dir + '/{}/img'.format(subdir_name) ):
        os.makedirs(output_dir + '/{}/img'.format(subdir_name))
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
    pred_df_concat = vad_predict(df, output_dir + '/' + subdir_name, \
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
    only_talk.plot()
    only_talk_plot = fname + '_talk_plot'
    plt.savefig(output_dir + '/{}/img/{}.jpg'.format(subdir_name, only_talk_plot))
    ### Turn taking matrix
    turn_taking_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='turn_taking')
    short_res_df = make_turn_taking_df(only_talk, dialog_len_list, n_person, tt_term=tt_term, mode='short_res')
    tt_and_short = turn_taking_df + short_res_df

    ### silece detect
    combi = combi_suspect(only_talk, n_person=n_person)
    sil_breaker = silence_breaker(only_talk, combi, n_person=n_person, s_term=s_term)
    s_breaker_df = silence_table(sil_breaker, n_person=n_person)

    ### mirroring_detection
    m_matrix = mirroring_matrix(dialog_len_list, n_person, m_term=m_term)

    ### turn taking matrix
    tt_mat, res_mat = turn_taking_matrix(only_talk, n_person=n_person, dialog_len_list=dialog_len_list, s_term=s_term, tt_term=tt_term)
    # 엑셀 파일 열기 w/ExcelWriter
    writer = pd.ExcelWriter(output_dir + '/{}/{}.xlsx'.format(subdir_name,fname), engine='xlsxwriter')
    # 시트별 데이터 추가하기
    dd_df.to_excel(writer, sheet_name= 'Prediction result', index=False)
    turn_taking_df.to_excel(writer, sheet_name= 'Count turn taking')
    short_res_df.to_excel(writer, sheet_name= 'Count short response')
    tt_and_short.to_excel(writer, sheet_name= 'Short and turn taking')
    s_breaker_df.to_excel(writer, sheet_name= 'Count silence breaker', index=False)
    m_matrix.to_excel(writer, sheet_name= 'Count mirriring')
    tt_mat.to_excel(writer, sheet_name= 'turn taking matrix', index=False)
    res_mat.to_excel(writer, sheet_name= 'response_matrix', index=False)

    # 엑셀 파일 저장하기
    writer.save()
