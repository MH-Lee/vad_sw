import pandas as pd
import itertools
from itertools import permutations, chain
from collections import Counter
import numpy as np


def remove(talk_column):
    """
    - input : 원본 데이터 프레임 dff의 'talk#' 컬럼
    - output : 'talk#' 컬럼이 -1-1-1-1-1-1  1 -1-1-1-1-1 인 경우 1을 -1로 만들어준 결과가 나옴

    - 아래 'post_smth' 함수에 쓰이는 함수
    - '-1'이 길게 진행되는 중에 1이 하나만 나오는 것은 에러일 가능성이 높다고 판단.
    """

    dest = []
    for i, v in enumerate(talk_column.values):
        if i == 0:
            pass
        elif(talk_column.values[i-1] + talk_column.values[i] == 0 and talk_column.values[i] + talk_column.values[i+1] == 0):
            if talk_column.values[i] == 1.0:
                dest.append(i)
    # print(dest) #어떤 인덱스에서 1 이 나오는가?
    for i in dest:
        talk_column.iloc[i, :] = talk_column.iloc[i, :].replace(1,-1)
    return talk_column


def post_smth(original_df, n_person):
    """
    - input : 원본 데이터프레임 dff
    - output : person1, person2, person3 컬럼 별로 remove 함수를 취한 결과를 concat하여 다시 데이터프레임으로 반환

    """
    post_sm_list = list()
    new_columns = list()
    for p in range(1, n_person+1):
        test_tmp = original_df['talk{}'.format(p)]
        new_test_tmp = pd.DataFrame(test_tmp).reset_index(drop = True)
        post_sm_list.append(remove(new_test_tmp))
        new_columns.append("talk{}".format(p))
    new = pd.concat(post_sm_list, axis = 1)
    original_df = original_df.drop(new_columns, axis = 1)
    new_df = pd.concat([original_df, new], axis = 1)
    return new_df


def detect_talk_break_length(talk):  #p1 의 talk1 열만 들어간다.
    '''
    - input: post smoothing이 끝난 new_df의 한 명에 해당하는 talk 데이터가 input으로 들어간다. ex. person1의 talk1 컬럼만 input으로 들어간다.
    - output: talk# 컬럼의 값을 1은 1대로 축적, -1은 -1대로 축적한 값을 리스트로 반환
        - 가령, 1111 이 나오면 4로 변환/ -1-1-1-1이 나오면 -4로 변환
        - 가령, 1111, -1-1-1, 1111 과 같이 양수, 음수 바뀔 때마다 값을 1 혹은 -1로 초기화하여 다시 축적한 값을 반환
        - ex. 111-1-111111 --> 3 -2 5 로 변환
    '''
    total = 0
    dialog_len = []
    for i in range(len(talk)): #0 - 50662
        prev_total = total
        total += talk[i]
        #if i >= 5445 and i <=5600:
            #print("id", i, "prev:", prev_total, "total", total)
        if(total < 0 and total > prev_total):  #-1에서 1로 넘어갔다면, 1로 초기화하고 축적
            dialog_len.append(prev_total)
            total = 1

        elif(total > 0 and total < prev_total): # 1에서 -1로 넘어갔다면, -1로 초기화하고 축적
            dialog_len.append(prev_total)
            total = -1
        if(i==len(talk)-1):
            dialog_len.append(total)# 마지막 누계는 무조건 append

    return dialog_len

def dup(guest_dup):
    """
    * 아래 'count_turn_takes' 함수에 쓰임

    - input : only_talk 에서 host의 음성 구간에 해당하는 guest의 talk# 컬럼
    - output : input에 해당하는 guest의 talk# 컬럼에서 나온 1의 합계

    --> host가 말하는 동안 guest는 말을 하지 않아야 하지만, guest가 간단한 미러링 하는 것은 봐준다. 미러링 도합 500 이하가 되도록 제한할 예정
    """
    a = pd.DataFrame(guest_dup)
    b = (a.iloc[:,0] == 1.0).sum()
    return b

def get_start_end_times(dialog_len_host, dialog_len_guest):
    """
    - input : detect_talk_break_length 결과값1, detect_talk_break_length 결과값2
    - output:
        * host의 음성구간의 시작점 인덱스, 끝점 인덱스
        * guest의 음성구간의 시작점 인덱스, 끝점 인덱스
        * guest의 음성구간의 길이 --> short-response 가려내기 위함

    - dialog_len_host 에서 dialog_len_guest 로의 턴테이킹만 측정
    - dialog_len_host가 언제나 먼저 말하는 역할. dialog_len_guest는 항상 dialog_len_host가 말한 다음에 말하는 역할
    """
    start = 0
    start_end = []
    if dialog_len_host[-1] < 0 : # 마지막 구간이 침묵 구간이라면
        for i in range(len(dialog_len_host) -1):
            start += abs(dialog_len_host[i])
            if dialog_len_host[i] < 0: # a: host 음성구간 시작 인덱스, b : host 음성구간 끝 인덱스
                a = start
                b = (start + abs(dialog_len_host[i + 1]) - 1)
                start_end.append([a,b])
    else: # 마지막 구간이 발화 구간이라면 모든 원소에 인덱싱을 해야함
        for i in range(len(dialog_len_host)):
            start += abs(dialog_len_host[i])
            if dialog_len_host[i] < 0: # a: host 음성구간 시작 인덱스, b : host 음성구간 끝 인덱스
                a = start
                b = (start + abs(dialog_len_host[i + 1]) - 1)
                start_end.append([a,b])


    start2 = 0
    start_end2 = []
    guest_talk_times = []
    if dialog_len_guest[-1] < 0 : # 마지막 구간이 침묵 구간이라면

        for i in range(len(dialog_len_guest)-1):  ##-1한것을 제거함
            start2 += abs(dialog_len_guest[i])
            if dialog_len_guest[i] < 0: # c: guest 음성구간 시작 인덱스, d : guest 음성구간 끝 인덱스
                c = start2
                d = (start2 + abs(dialog_len_guest[i + 1]) - 1)
                start_end2.append([c,d])

            else:
                guest_talk_times.append(dialog_len_guest[i])

    #######
    else:      # 마지막 구간이 발화 구간이라면
        for i in range(len(dialog_len_guest)):  ##-1한것을 제거함
            start2 += abs(dialog_len_guest[i])
            if dialog_len_guest[i] < 0: # c: guest 음성구간 시작 인덱스, d : guest 음성구간 끝 인덱스
                c = start2
                d = (start2 + abs(dialog_len_guest[i + 1]) - 1)
                start_end2.append([c,d])

            else:
                #print(dialog_len_guest[i])
                guest_talk_times.append(dialog_len_guest[i])


    return start_end, start_end2, guest_talk_times


def count_turn_takes(only_talk, host_number, guest_number, host_start_end, guest_start_end, guest_talk_times1, howlong=300): #숏 리스펀스
    """
    *input:
        - host_number = host가 되는 person의 인덱스 => p1 p2 p3 == 0 1 2
        - guset_number = guest가 되는 person의 인덱스 => p1 p2 p3 == 0 1 2
        - host_start_end = get_start_end_times 함수 결과의 start_end (host 음성 구간의 시작 인덱스, 끝 인덱스)
        - guest_start_end = get_start_end_times 함수 결과의 start_end2 (guest 음성 구간의 시작 인덱스, 끝 인덱스)
        - guest_talk_times1 = get_start_end_times 함수 결과의 guest_talk_times
        - howlong = short response 길이 상한선

    * output:
        - total_responses : 300 초과하는 구간 동안 말했을 경우만 turn taking 카운팅
        - short_responses : 300 이하의 짧은 음성 구간도 turn taking으로 간주하여 카운팅
        - change : total_responses 및 short_responses에 대하여 turn taking이 언제 일어났는지 인덱스 반환

    """
    short_responses = 0
    total_responses = 0
    change = []
    ######
    turn_taking_matrix = {str(host_number+1) + '->' + str(guest_number+1) + ' start': []}
    columns = list(turn_taking_matrix)

    response_matrix = {str(host_number+1) + '->' + str(guest_number+1) + ' start': []}
    columns2 = list(response_matrix)
    ######
    # print("host : ", host_number)
    # print("guest: ", guest_number)
    for i in range(len(host_start_end)):
        k = 0    #i 는 host의 인덱스, k는 guest의 인덱스
        while k<=len(guest_start_end)-1 and guest_start_end[k][0] < host_start_end[i][1]:   #k +=1 결과가 len(guest_start_end)보다는 작거나 같아야 한다.  AND guest 의 시작점이 host의 끝점 인덱스보다 작은 경우에는 k에 +1 을 하여, guest_start_end 결과의 그 다음 인덱스로 넘어가도록 한다.
            k += 1   # i를 돌면서 k를 업데이트, 각 i 별로 +1 / guest 의 시작점이 host의 끝점 인덱스보다 작은 경우에 +1 을 하여 다음 인덱스로 넘어간다.

        #print(i, k,host_start_end[i], guest_start_end[k])
        if k<=len(guest_start_end)-1 and guest_start_end[k][0] -host_start_end[i][1] <= 300 and  guest_start_end[k][0] -host_start_end[i][1] > 0: # 디폴트: 300 이내에 말 해야함 AND guest 의 시작점이 host의 끝점 인덱스보다 커야함
            ## host 가 말하는 중에는 guest가 침묵하고 있어야함. 다만, total 500 정도 말한 것은, 호응/미러링일 수 있으므로, 이 경우는 턴테이킹
            if dup(only_talk.iloc[int(host_start_end[i][0]): int(host_start_end[i][1])+1, guest_number]) <= 500:
                ## guest 가 말하는 중에는 host가 침묵하고 있어야함. 다만, total 500 정도 말한 것은, 미러링일 수 있으므로, 턴테이킹으로 분류
                if dup(only_talk.iloc[int(guest_start_end[k][0]): int(guest_start_end[k][1])+1, host_number]) <= 500:
                    ### host의 말길이도 300 이상이어야함
                    if host_start_end[i][1] - host_start_end[i][0] > 300:
                        # print(guest_talk_times1)
                        # print(k)
                        if guest_talk_times1[k] <= howlong:
                            short_responses += 1
                            change.append(k)
                            ## idx 및 데이터프레임을 출력할 수 있어야함. 딕셔너리 형태로
                            ## {'A -> B', start, end}
                            ## k 전의 host의 말의 끝 index도 알면 좋겠다. --> i 번째 말의 끝점 인덱스 반환 (표에는 이 인덱스 기준으로 어팬드)
                            ## k: guest의 말의 시작 --> k 의 인덱스를 반환
                            values2 = [guest_start_end[k][0]]
                            zipped2 = zip(columns2, values2)
                            a_dictionary2 = dict(zipped2)
                            response_matrix = pd.DataFrame(response_matrix).append(a_dictionary2, True)
                            ####
                        else:
                            total_responses += 1
                            change.append(k)
                            ## idx 및 데이터프레임을 출력할 수 있어야함. 딕셔너리 형태로
                            ## k: guest의 말의 시작 -->  k의 인덱스를 반환
                            ## guest 의 말의 끝점 인덱스도 반환
                            values = [guest_start_end[k][0]] # 게스트의 시작점은 맞음
                            zipped = zip(columns, values)
                            a_dictionary = dict(zipped)
                            turn_taking_matrix = pd.DataFrame(turn_taking_matrix).append(a_dictionary, True)
                            ####
    return total_responses, short_responses, turn_taking_matrix, response_matrix


def count_tt_detail(only_talk, dialog_len_list, n_person, tt_term):
    detail_result = list()
    for pm in permutations(range(0,n_person),2):
        host_end, guest_start, guest_talk_times = get_start_end_times(dialog_len_list[pm[0]], dialog_len_list[pm[1]])
        # count_result : [total_res(int), short_res(int), change(list)]
        count_result = count_turn_takes(only_talk, \
                                        pm[0], pm[1], \
                                        host_end, \
                                        guest_start, \
                                        guest_talk_times,
                                        howlong=tt_term)
        detail_result.append(count_result)
    return detail_result


def make_turn_taking_df(only_talk, dialog_len_list, n_person, mode='turn_taking', tt_term=50):
    """
    - input : detect_talk_break_length 함수에 talk1, talk2, talk3 을 넣고 순서대로 input으로 사용
    - output : Short response 없이 total_responses 에 대한 턴테이킹만 카운팅

    """
    mode_dict = {'turn_taking':0, 'short_res':1}
    detail_result = count_tt_detail(only_talk, dialog_len_list, n_person, tt_term=tt_term)
    new_columns = ['to_p{}'.format(i) for i in range(1, n_person+1)]
    np_data = np.zeros((n_person, n_person))
    for idx, pm in enumerate(permutations(range(0, n_person), 2)):
        # print(pm)
        # print(pm[0]+1, "->", pm[1]+1)
        np_data[pm] = detail_result[idx][mode_dict[mode]]
    np_data = np_data + np.diag(np_data.sum(axis=1))
    df = pd.DataFrame(np_data, columns=new_columns, index=new_columns)
    return df

##### Silence Check
def combi_suspect(df, n_person):
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
    df_sum_row = df.sum(axis=1)
    for i in range(len(df)):
        if (df_sum_row[i]) == -1* int(n_person):  ### 총합이 -3일때 침묵인 상황
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

def silence_breaker(df, combi, n_person, s_term=300):
    """
    - combi : the result value of combi_suspect function
    - s_term : the time period during silence

    - returns person indices who breaks a silence
    - person index : person1 --> 0, person2 --> 1, person3 -->2

    """
    comb_silence_values = combi.values

    after_silence_person = []
    before_silence_person =[]

    after_silence = []
    before_silence = []

    ##추가
    start_break_idx = []
    end_break_idx = []
    idx_pair = []
    ##
    select_columns = ['talk{}'.format(i) for i in range(1, n_person+1)]
    for i, value in enumerate(comb_silence_values):
        if(comb_silence_values[i,0] == 0):
            pass
        elif(comb_silence_values[i,0] - comb_silence_values[i-1,0] >= 2): #침묵이 깨졌을 떄
            #print(comb_silence_values[i-1,1])
            if(comb_silence_values[i-1,1] <= -s_term): #침묵이 300 이상 진행 되었을 떄,


                break_idx = comb_silence_values[i-1,0] + 1 # 침묵을 깬 사람이 발생한 row의 index
                til_break_df = df[:break_idx + 1].loc[:,('talk1', 'talk2', 'talk3')] # 첫 row 부터 침묵을 깬 row 까지 데이터프레임 슬라이스
                # 만일 til_break_df 에서 row의 원소들의 총합이 -3이 아닌 경우가 단 하나만 있다면, 해당 break_idx는 밑에서 어팬드 하지 않고 pass
                til_break_df['sum'] = til_break_df.apply(lambda row: row.talk1 + row.talk2 + row.talk3, axis = 1)


                # 가장 처음 발생한 breaker는 silence breaker가 아니라, 단순히 처음 대화를 시작한 경우일 뿐이므로 제외
                if len(np.where(til_break_df['sum'] != -3)[0]) >= 2:

                    ## 최신_추가##
                    start_break_idx.append(break_idx) # breaker가 말한 구간의 시작 행

                    end_break = comb_silence_values[i,0] - 1 # breaker가 말한 구간의 마지막 행
                    end_break_idx.append(end_break)


                    #after_silence.append(break_idx) # 침묵이 300 이상 진행 되었을때, 그것이 깨진 행을 append한다.

                    ###########추가한 부분#############################
                    # before_silence 리스트에 값 추가
                    ## silence 직전에 누군가 말을하고 있었던 행의 인덱스 추가
                    acc_idx = abs(comb_silence_values[i-1,1] + 1)
                    real_idx = comb_silence_values[i-1,0] - acc_idx - 1
                    before_silence.append(real_idx)
                    ###################################################


    ## 최신 추가####################################################
    # 1) 따로 구분 되어있는 start_break_idx, end_break_idx 리스트의 원소를 순서쌍으로 하는 리스트를 만든다.
    # idx_pair = [[start_break[0], end_break[0]], [start_break[1], end_break[1]], [start_break[2], end_break[2]]....]
    for i, v in enumerate(start_break_idx):
        idx = end_break_idx[i] #breaker의 마지막 행의 인덱스
        #print([v, idx])
        idx_pair.append([v, idx]) #[breaker의 첫 행 인덱스/ breaker의 마지막 행의 인덱스] 순서쌍

    for i in idx_pair:
        row = [df['talk1'][i[0]],df['talk2'][i[0]], df['talk3'][i[0]]]
        row1 = [i for i in range(len(row)) if row[i] == 1.0] ## ex. [-1, 1, 1] 중에서 1,1에 대응되는 person을 구해라
        if len(row1) == 1: # a) breaker가 한 명일 때는 그 인덱스를 바로 반환
            row1_value = int(str(row1)[1:-1])
            after_silence_person.append([i[0],row1_value]) #### [breaker 첫 행 인덱스, breaker person 인덱스]
        else:
            # b) breaker가 두 명 이상일 때는 가장 많이 말한 경우만 반환
            person = [df[i[0]:i[1]].sum()[0], df[i[0]:i[1]].sum()[1], df[i[0]:i[1]].sum()[2]]
            # only_talk의 person 별로 [start_break], [end_break] 인덱스 구간에서의 1 or -1 모두 더한 값이 가장 클 때, 말을 가장 많이함
            max_idx = [i for i in range(len(person)) if max(person) == person[i]][0]
            after_silence_person.append([i[0], max_idx]) #### [start_breaker, 가장 많이 말한 person의 인덱스]


    ####최신_추가#########################################################
    # 침묵 직전에 누가 말했나? : 침묵 직전에 2명 이상이 말했을 수 있으므로, 해당하는 모든 인덱스 반환하도록 수정
    # before_silence = 침묵 직전에 누군가가 말을 한 행의 인덱스를 모두 어팬드한 리스트
    for k, j in enumerate(before_silence):
        row = [df['talk1'][j],df['talk2'][j], df['talk3'][j]]  #
        row1 = [i for i in range(len(row)) if row[i] == 1.0]
        before_silence_person.append([j, row1]) ## 침묵 직전에 발언한 사람이 2명이고 그 행이 153 --> [153, [0,1]]

    return before_silence_person, after_silence_person

def silence_table(whois, n_person):
    """
    - input : 아래 두 함수의 결과 --> breaker(combi_suspect(df), 400) 를 input으로 함
    - output: person 별로 침묵을 깬 횟수 카운팅

    creating a final table for silence breaker
    """
    # z = list(chain(*whois))
    if len(whois[1]) == 0:
        return pd.DataFrame()
    else:
        z = list(np.array(whois[1])[:,1])
        print("z:", z)
    df = pd.DataFrame.from_dict(Counter(z), orient='index').reset_index()
    df.columns = ['person', 'interruption']
    df.sort_values(by='person', ascending=True, inplace = True)
    df_final = df.copy()
    for p in range(n_person):
        df_final.loc[:, 'person'].replace(p, 'person{}'.format(p+1), inplace = True)
    return df_final.reset_index(drop=True)

#### add
def table_before(breaker_result):
    """
    - input : breaker함수의 결과값. ex. breaker(combi_suspect(df), 400)
    - output : 컬럼이 2개인 데이터프레임 반환
        - 1번째 열: silence 직전에 말했을 때의 인덱스
        - 2번째 열: silence 직전에 말한 사람
    """
    if len(breaker_result[0]) == 0:
        before_s = pd.DataFrame()
    else:
        texts = []
        before = pd.DataFrame(breaker_result[0])
        for i, v in enumerate(before.iloc[:,1]):
            person_idx = [x+1 for x in v]
            # print(person_idx)
            text = str(person_idx)[1:-1] + "->silence"
            # print(text)
            texts.append(text)

        before['person'] = texts
        before.rename(columns = {0 : 'order'}, inplace = True)
        before_s = pd.concat([before['order'], before['person']], axis = 1)
    return before_s

def table_after(breaker_result):
    """
    - input : breaker함수의 결과값. ex. breaker(combi_suspect(df), 400)
    - output : 컬럼이 2개인 데이터프레임 반환
        - 1번째 열: silence 직후에 말했을 때의 인덱스
        - 2번째 열: silence 직후에 말한 사람
    """
    if len(breaker_result[1]) == 0:
        after_s = pd.DataFrame()
    else:
        texts = []
        after = pd.DataFrame(breaker_result[1])
        for i, v in enumerate(after.iloc[:, 1]):
            person_idx = v + 1
            #print(person_idx)
            text = "silence->" + str(person_idx)
            #print(text)
            texts.append(text)

        after['person'] = texts
        after.rename(columns = {0:'order'}, inplace = True)
        after_s = pd.concat([after['order'], after['person']], axis = 1)
    return after_s

### turn_taking matrix
def add_label(turn_matrix):
    """
    - 아래 dict_to_df 함수의 내장 함수
    - input : total_turn의 각 dataframe
    - output : total_turn의 각 dataframe마다 p1->p2, p3->p1 과 같은 정보를 컬럼으로 추가한 것을 반환
    """
    if len(turn_matrix.columns) == 1:
        turn_matrix['person'] = str(turn_matrix.columns)[8:18]
        turn_matrix[:]
    return turn_matrix[:]

def dict_to_df(total):
    """
    - input : total_turn (turn_taking이 어떤 인덱스에서 일어났는지, 누가 일으켰는지에 대한 정보를 담은 데이터프레임)
    - output :
        total_turn의 각dataframe별로 p1->p2, p2->p3과 같은 정보를 컬럼으로 추가함
    """
    total_turn2 = []
    # dict --> dataframe으로 바꾼다.
    for i in total:
        if type(i) == dict:
            i = pd.DataFrame(i)
            total_turn2.append(i)
        else:
            total_turn2.append(i)

    # total의 각 요소(데이터프레임)에 컬럼을 추가한다.
    for k in total_turn2:
        add_label(k)
        total = total_turn2[:]

    if len(total[0].columns) == 2:
        for j in total:
            j.columns = ['order', 'person']
    return total

def turn_sequence(turn,before, after):
    """
    - input : 순서대로 --> total_turn, before_table 함수 결과, after_table 함수 결과

    - output :
        - p1-> p2과 같은 정보를 담은 컬럼이 추가된 total_turn의 각 데이터 프레임을 아래로 모두 컨켓.
        - 위의 데이터프레임에서 index를 뜻하는 첫 번째 컬럼을 기준으로 sorting
    """
    concat = []
    for i, v in enumerate(dict_to_df(turn)):
        concat.append(dict_to_df(turn)[i])

    turn_taking = pd.concat(concat, axis = 0).reset_index(drop = True)
    total = pd.concat([turn_taking, after, before], axis = 0)
    total_sort = total.sort_values('order')
    return total_sort.reset_index(drop = True)

def short_sequence(short):
    """
    - input : total_response 함수 결과
    - output :
        - p1-> p2과 같은 정보를 담은 컬럼이 추가된 input 각 데이터 프레임을 아래로 모두 컨켓.
        - 위의 데이터프레임에서 index를 뜻하는 첫 번째 컬럼을 기준으로 sorting
    """
    concat2 = []
    for i, v in enumerate(dict_to_df(short)):
        concat2.append(dict_to_df(short)[i])

    short = pd.concat(concat2, axis = 0).reset_index(drop = True)
    total_sort = short.sort_values('order')

    return total_sort.reset_index(drop = True)

#### mirroring dialog_detection
def mirroring(host1, guest1, m_term = 350):
    """
    - 아래 mirroring_matrix 의 내장 함수로 쓰임
    - input : detect_talk_break_length 의 결과값 중 두 개만 pair로 들어감 : 첫 번째로 들어가는 결과값이 host1, 그 다음에 들어가는 결과값이 guest1
    - output : host1의 음성구간 동안 guest1 의 미러링 횟수를 카운팅

    """
    mirror2 = 0
    for i in range(len(get_start_end_times(host1, guest1)[0])):# 호스트 기준으로만 미러링을 센다
        for k in range(len(get_start_end_times(host1, guest1)[1])):
            # host의 시작점 보다 guest의 시작점 index가 커야함 & host 의 끝점보다 guest의 시작점 index가 작아야함
            if (get_start_end_times(host1, guest1)[0][i][0] < get_start_end_times(host1, guest1)[1][k][0]) and (get_start_end_times(host1, guest1)[0][i][1] > get_start_end_times(host1, guest1)[1][k][0]):
                # host의 시작점보다 guset의 끝점 index가 커야함  & host의 끝점보다 guest의 끝점 index가 작아야함
                if (get_start_end_times(host1, guest1)[0][i][0] < get_start_end_times(host1, guest1)[1][k][1]) and (get_start_end_times(host1, guest1)[0][i][1] > get_start_end_times(host1, guest1)[1][k][1]):
                    if get_start_end_times(host1, guest1)[1][k][1] - get_start_end_times(host1, guest1)[1][k][0] <= m_term:
                        mirror2 += 1
    return mirror2

# 포루프 돌려야 할 대상 : dialog_len1, dialog_len2, dialog_len3
def mirroring_matrix(dialog_len_list, n_person, m_term = 350):
    """
    input : detect_talk_break_length 결과를 talk1 talk2 talk3 순서 대로 넣어준다.
    output : mirroring 횟수를 나타내는 데이터프레임 출력

    """
    new_columns = ['to_p{}'.format(i) for i in range(1, n_person+1)]
    np_data = np.zeros((n_person, n_person))
    for idx, pm in enumerate(permutations(range(0, n_person), 2)):
        # print(pm)
        # print(pm[0]+1, "->", pm[1]+1)
        np_data[pm] = mirroring(dialog_len_list[pm[0]],\
                                dialog_len_list[pm[1]],\
                                m_term)
    mirroring_df =pd.DataFrame(np_data, columns=new_columns, index=new_columns)
    return mirroring_df

### turn taking matrix
def turn_taking_matrix(only_talk, n_person, dialog_len_list, s_term, tt_term):
    combi = combi_suspect(only_talk, n_person=n_person)
    sil_breaker = silence_breaker(only_talk, combi, n_person=n_person, s_term=s_term)
    s_breaker_df = silence_table(sil_breaker, n_person=n_person)
    silence_after = table_after(sil_breaker)
    silence_before = table_before(sil_breaker)
    detail_result = count_tt_detail(only_talk, dialog_len_list, n_person, tt_term=tt_term)
    total_turn = [res[2] for res in detail_result]
    total_short = [res[3] for res in detail_result]
    ts_df = turn_sequence(total_turn, silence_before, silence_after)
    ts_short = short_sequence(total_short)
    return ts_df, ts_short
