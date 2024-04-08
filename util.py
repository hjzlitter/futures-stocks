import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def adf_test(timeseries):
    # print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dfoutput

def johansen_cointegration_test(df, det_order = -1, k_ar_diff = 1):
    """Perform Johansen cointegration test on dataframe df
    det_order : int
        * -1 - no deterministic terms
        *  0 - constant term
        *  1 - linear trend
    k_ar_diff : int, number of lagged differences in the model.
    """
   
    result = coint_johansen(df, det_order, k_ar_diff)
    # print('Results of Johansen Cointegration Test:')
    # print(f"Test statistic: {result.lr1}")
    # print(f"Critical values: {result.cvt}")
    # print(f"Eigenstatistics: {result.lr2}")
    # print(f"Eigenvalues: {result.eig}")
    return result

# 假设你有一个名为df的Pandas DataFrame时间序列数据集，每列是一个时间序列
# johansen_cointegration_test(df)


# 根据fut 跟 stock 的df ，以及feature。生成需要的数据
def data_generator(fut_df, stock_df, feature='close'):
    # fut = 'CU9999'
    # stock = '600362'
    # print(fut, stock)
    # au = pd.read_csv(f"futures/{fut}.csv")
    # print(len(au))
    # au['date'] = pd.to_datetime(au['date'])
    # au['ChangeRatio'] = au['close'].diff() / au['close'].shift(1) 
    # au['ChangeRatio'].iloc[0] = 0
    
    
    # au = au[au.date>='2019-05-01']
    #stock_df = stock_read(fut)
    # print(len(stock_df[1]))
    # sig_col = []
    # for idx, df in enumerate(stock_df):
    # stock_df['date'] = pd.to_datetime(stock_df['date'])
    # df = df[df.date>='2019-05-01']
    # df.rename(columns={"Trddt": "date"}, inplace=True)
    merged_df = pd.concat([fut_df['date'], stock_df['date']])
    unique_times = merged_df.drop_duplicates()
    unique_times[~unique_times.isin(fut_df['date']) | ~unique_times.isin(stock_df['date'])]
    # p_value_list = []

    unique_times_df1 = set(fut_df['date'])
    unique_times_df2 = set(stock_df['date'])
    common_times = list(unique_times_df1.intersection(unique_times_df2))
    df1_common = fut_df[fut_df['date'].isin(common_times)]
    df2_common = stock_df[stock_df['date'].isin(common_times)]
    merged_df = pd.merge(df1_common, df2_common, on='date', suffixes=('_df1', '_df2'))
    data = merged_df[[f'{feature}_df1', f'{feature}_df2']]
    return data

def significant(data):
    p_value_list = []
    num_days = len(data)
    window_size = 30
    for i in range(num_days - window_size + 1):
        window_start = i
        window_end = i + window_size
        window_data = data.iloc[window_start:window_end]
        model = VAR(window_data)
        results = model.fit()
        # print(results.summary())
        p_values = results.pvalues    
        p_value_list.append(np.array(p_values.iloc[:,1]))
    significant_columns = [[i, arr] for i, arr in enumerate(p_value_list) if all(arr < 0.10)]
    return significant_columns


if __name__ == '__main__':
    pass