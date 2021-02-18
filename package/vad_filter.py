import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def savitzky_golay(df) :
    signal = df['signal']
    sa10 = pd.DataFrame(savgol_filter(signal, 11, 1))
    sa50 = pd.DataFrame(savgol_filter(signal, 51, 1))
    sa100 = pd.DataFrame(savgol_filter(signal, 101, 1))
    sa200 = pd.DataFrame(savgol_filter(signal, 201, 1))
    sa300 = pd.DataFrame(savgol_filter(signal, 301, 1))

    df_sa = pd.concat([sa10, sa50, sa100, sa200, sa300], axis = 1)
    df_sa.columns = ["sa10", 'sa50', 'sa100', 'sa200', 'sa300']
    return df_sa

def savitzky_golay_quadratic(df) :
    signal = df['signal']
    saq10 = pd.DataFrame(savgol_filter(signal, 11, 2))
    saq50 = pd.DataFrame(savgol_filter(signal, 51, 2))
    saq100 = pd.DataFrame(savgol_filter(signal, 101, 2))
    saq200 = pd.DataFrame(savgol_filter(signal, 201, 2))
    saq300 = pd.DataFrame(savgol_filter(signal, 301, 2))

    df_saq = pd.concat([saq10, saq50, saq100, saq200, saq300], axis = 1)
    df_saq.columns = ["saq10", 'saq50', 'saq100', 'saq200', 'saq300']
    return df_saq

def moving_average(df) :
    signal = df['signal']
    ma10 = signal.rolling(10, min_periods = 1).mean().to_frame()
    ma50 = signal.rolling(50, min_periods = 1).mean().to_frame()
    ma100 = signal.rolling(100, min_periods = 1).mean().to_frame()
    ma200 = signal.rolling(200, min_periods = 1).mean().to_frame()
    ma300 = signal.rolling(300, min_periods = 1).mean().to_frame()

    df_ma = pd.concat([ma10, ma50, ma100, ma200, ma300], axis = 1)
    df_ma.columns = ["ma10", 'ma50', 'ma100', 'ma200', 'ma300']
    df_ma = df_ma.fillna(0)
    return df_ma

def moving_standard(df) :
    signal = df['signal']
    sd10 = signal.rolling(10, min_periods = 1).std().to_frame()
    sd50 = signal.rolling(50, min_periods = 1).std().to_frame()
    sd100 = signal.rolling(100, min_periods = 1).std().to_frame()
    sd200 = signal.rolling(200, min_periods = 1).std().to_frame()
    sd300 = signal.rolling(300, min_periods = 1).std().to_frame()

    df_sd = pd.concat([sd10, sd50, sd100, sd200, sd300], axis = 1)
    df_sd.columns = ["sd10", 'sd50', 'sd100', 'sd200', 'sd300']
    df_sd = df_sd.fillna(0)
    return df_sd

def gaussian_fiter(df) :
    signal = df['signal']
    windows_size = [10, 50, 100, 200, 300]
    for m in windows_size :
        globals()['ga{}'.format(m)] = pd.DataFrame(gaussian_filter1d(signal, sigma =((m-1)/(2*2.5))))

    df_ga = pd.concat([ga10, ga50, ga100, ga200, ga300], axis = 1)
    df_ga.columns = ["ga10", 'ga50', 'ga100', 'ga200', 'ga300']
    #df_ga = df_ga.fillna(0)
    return df_ga

# 위의 filter로 생성한 변수들을 기존 데이터프레임에 추가
def feature_generation(df) :
    df_sa = savitzky_golay(df)
    df_saq = savitzky_golay_quadratic(df)
    df_ma = moving_average(df)
    df_sd = moving_standard(df)
    df_ga = gaussian_fiter(df)

    final_df = pd.concat([df, df_ga, df_sa, df_saq, df_ma, df_sd], axis = 1)
    final_df = final_df[['person', 'signal',
                         "ga10", 'ga50', 'ga100', 'ga200', 'ga300',
                         "sa10", 'sa50', 'sa100', 'sa200', 'sa300',
                         "saq10", 'saq50', 'saq100', 'saq200', 'saq300',
                         "ma10", 'ma50', 'ma100', 'ma200', 'ma300',
                         "sd10", 'sd50', 'sd100', 'sd200', 'sd300']]
    return final_df
