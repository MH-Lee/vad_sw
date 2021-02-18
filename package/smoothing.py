import pandas as pd
import numpy as np
from tqdm import tqdm

def assign_label(value):
    if value >= 0.8:
        label = 1
    elif (value < 0.8) & (value > 0.5):
        label = 0.5
    else:
        label = 0
    return label


def smoothing(pre_smth):
    smth = np.array(pd.Series(pre_smth).apply(assign_label), dtype=np.float)  
    new_smth = smth.copy()           #np.unique(new_smth) = 0, 0.5, 1
    new_smth = pd.DataFrame(new_smth)
    new_smth.columns = ['new_smth']
    voice = new_smth[new_smth['new_smth']==1] # voice --> 데이터가 1인 것만 모아서 데이터 프레임을 만든다. 
    for i in tqdm(range(0, len(new_smth))):  
        
        if i == 0: # 첫 번째 인덱스는 지나간다. 
            continue
        elif i == len(smth)-1: # 마지막 인덱스는 지나간다.
            continue
        elif i <= int(voice.index[0]): # 현재 인덱스가 voice 의 첫 번째 인덱스보다 작거나 같으면 지나간다.  
            continue
        elif i >= int(voice.index[-1]): # 현재 인덱스가 voice 의 마지막 인덱스보다 크거나 같으면 지나간다.         
            continue
        else: 
            a = i - voice.loc[:i-1,:].index[-1] # (i) - (i 발생 전의 voice 중 가장 마지막 인덱스)
            b = voice.loc[i+1:,:].index[0] - i  # (i 이후 voice 중 가장 첫 번째 인덱스) - (i)
            if (a < 200) & (b < 200) & (a+b < 200): ## 위의 a,b 구간이 충분이 작아야 1로 변경해준다. 
                new_smth.loc[i,'new_smth'] = 1
                
    semi_voice = new_smth[new_smth['new_smth']==0.5]
    for j in tqdm(semi_voice.index):
    
        if (len(new_smth.loc[j - 200 : j:200,:][new_smth.loc[j - 200 : j:200,:]['new_smth']==1])>=1):
            new_smth.loc[j,'new_smth']=1
        else:
            new_smth.loc[j,'new_smth']=0
        
    new_voice = new_smth[new_smth['new_smth']==1]
    for i in tqdm(range(0, len(new_smth))):  #smth : 원본 
        if i == 0:
            continue
        elif i == len(smth)-1:
            continue
        elif i <= int(voice.index[0]):
            continue
        elif i >= int(voice.index[-1]):        
            continue
        else:
            a = i - new_voice.loc[:i-1,:].index[-1]
            b = new_voice.loc[i+1:,:].index[0]-i
            if (a < 200) & (b < 200) & (a+b < 200):
                new_smth.loc[i,'new_smth'] = 1
    return new_smth


def turn_taking(pred): ## 스무딩 완료한 것을 넣어야함 
    a = pred['new_smth'][1:]
   
    turn_taking = 0
    for i, value in enumerate(a):
        if value - a.iloc[i-1] == -1:
            turn_taking += 1
            
    return turn_taking


# 모든 traing set의 turn taking 개수를 세고 원본과 비교

def total_turn_taking (train, true) :
    train_tt = turn_taking(train)
    true_tt = turn_taking(true)
    total = [train_tt, true_tt]
    df_total = pd.DataFrame(total, index = ['train','true_test'])
    df_total.columns = ['turn_taking']
    return df_total

