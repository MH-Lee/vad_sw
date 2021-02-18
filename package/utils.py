import pandas as pd
import numpy as np
from itertools import islice

# 데이터 정규화
def normalization(df) :
    # if n_person == 3:
    #     x_df = df[['x1', 'x2', 'x3']]
    # else:
    #     x_df = df[['x1', 'x2']]
    n_person = df.shape[1]
    df_new_columns = ['x{}'.format(i) for i in range(1, n_person+1)]
    x_df = df[df_new_columns]
    normalized_x = (x_df - x_df.mean()) / x_df.std()
    normalized_df = pd.DataFrame(normalized_x)
    return normalized_df


# 각 시나리오 file 별로 x1, x2, x3를 행단위로 이어준다.
# 데이터 형태 :
# --------------
# | [x1] |
# | [x2] |
# | [x3] |
# --------------
def infile_concat(df) :
    total_list = list()
    for i in range(df.shape[1]):
        person_x = pd.concat([(i+1)*pd.Series(np.ones(pd.Series(df['x{}'.format(i+1)]).shape[0])), pd.Series(df['x{}'.format(i+1)])], axis = 1)
        total_list.append(person_x)
    total = pd.concat(total_list, axis = 1)

    a_list = list()
    b_list = list()
    for i in range(df.shape[1]):
        a_list.append(total.iloc[:, 2*i])
        b_list.append(total.iloc[:, 2*i+1])
    a = pd.concat(a_list)
    b = pd.concat(b_list)
    aa = pd.DataFrame(a).reset_index(drop=True)
    aa.columns = ['idx']
    bb = pd.DataFrame(b).reset_index(drop=True)
    bb.columns = ['signal']

    total_x = pd.concat([aa,bb], axis = 1).reset_index(drop=True)
    infile_concated_df = pd.DataFrame(total_x)
    infile_concated_df.columns = ['person', 'signal']
    infile_concated_df = infile_concated_df.reset_index(drop=True)
    return infile_concated_df

# 절댓값 취하기
def absolute(df) :
    absolute_df = np.abs(df)
    return absolute_df
