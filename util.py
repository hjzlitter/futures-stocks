import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from data import pair, fut_list, fut_read, stock_read
from sklearn.metrics import r2_score


def adf_test(timeseries):
    # print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)
    return dfoutput

def EG(df):
    X_t = df.iloc[:,0]
    Y_t = df.iloc[:,1]
    X = sm.add_constant(X_t)  # 添加常数项
    model = sm.OLS(Y_t, X).fit()
    coefficients = model.params
    t_values = model.tvalues
    p_values = model.pvalues
    r_squared = model.rsquared
    result_df = pd.DataFrame({'Coefficients': coefficients, 't-values': t_values, 'p-values': p_values ,'r_squared': r_squared})
    residuals = model.resid
    dfoutput = adf_test(residuals)
    return result_df, dfoutput

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
    # merged_df = pd.concat([fut_df['date'], stock_df['date']])
    # unique_times = merged_df.drop_duplicates()
    # unique_times[~unique_times.isin(fut_df['date']) | ~unique_times.isin(stock_df['date'])]
    # # p_value_list = []

    # unique_times_df1 = set(fut_df['date'])
    # unique_times_df2 = set(stock_df['date'])
    # common_times = list(unique_times_df1.intersection(unique_times_df2))
    # df1_common = fut_df[fut_df['date'].isin(common_times)]
    # df2_common = stock_df[stock_df['date'].isin(common_times)]
    # merged_df = pd.merge(df1_common, df2_common, on='date', suffixes=('_df1', '_df2'))
    # data = merged_df[[f'{feature}_df1', f'{feature}_df2']]
    # label = merged_df['label']
    data = pd.merge(fut_df, stock_df, how='outer', left_on='date', right_on='date', suffixes=('_Future', '_Stock'))
    for column in data.columns[15:-1]:
        data[column] = data[column].fillna(data[column].rolling(window=5, min_periods=1).mean())#.fillna(method='ffill') #.
        data[column] = data[column].fillna(data[column].rolling(window=5, min_periods=1).mean())
        data[column] = data[column].fillna(data[column].rolling(window=5, min_periods=1).mean())
        data[column] = data[column].fillna(data[column].rolling(window=5, min_periods=1).mean())
        data[column] = data[column].fillna(data[column].rolling(window=5, min_periods=1).mean())
    data = data[data.date>='2019-06-01']
    condition = data['ChangeRatio_Stock'] > 0
    label = np.where(condition, True, data['label'])
    # print([f'{feature}_df1', f'{feature}_df2'])
    # print(data.columns)
    return data[[f'{feature}_Future', f'{feature}_Stock']], label

def significant(data, label=None):
    p_value_list = []
    acc_list = []
    num_days = len(data)
    window_size = 30
    for i in range(num_days - window_size + 1):
        window_start = i
        window_end = i + window_size
        window_data = data.iloc[window_start:window_end]
        model = VAR(window_data)
        results = model.fit()
        if label:
            cnt = 0
            for j in range(window_start, window_end-1):
                forcast = results.forecast(data.values[j].reshape(1, -1), steps=1)
                if(label[j+1]==(forcast[0][1]>0)):
                    cnt += 1 
            acc = cnt/window_size   
            acc_list.append(acc)
            # print(results.summary())
        # print(results.summary())
        p_values = results.pvalues    
        p_value_list.append(np.array(p_values.iloc[:,1]))
    significant_columns = [[i, arr] for i, arr in enumerate(p_value_list) if all(arr < 0.10)]
    return significant_columns, acc_list

def para(significant_idx, data, label=None):
    window_size = 30
    window_start = significant_idx
    window_end = significant_idx + window_size
    window_data = data.iloc[window_start:window_end]
    model = VAR(window_data)
    results = model.fit()


def benefits_show(significant_idx, data, idx, fut_df, stock_df, feature='close'):
    # significant_idx
    window_size = 30
    window_start = significant_idx
    window_end = significant_idx + window_size
    window_data = data.iloc[window_start:window_end]
    model = VAR(window_data)
    results = model.fit()
    fut_df = fut_df[fut_df.date>='2019-06-01']
    df = stock_df[idx]
    df = df[df.date>='2019-06-01']
    print(len(df))
    forecast_list = []
    for j in range(window_start, window_end - 1):
        forecast, lower, upper = results.forecast_interval(data.values[j].reshape(1, -1), steps=1, alpha=0.05)
        forecast_list.append(forecast[0][0])
    if feature == 'close': 
        stock_precict = np.array(forecast_list)
        print_feature = 'Price'
        plt_data = pd.DataFrame({
            'Date': fut_df.date[window_start+1:window_end],
            'Stock_Predict': stock_precict,
            'Stock_Truth': window_data.iloc[1:, 1].values,
            'Future_Price': window_data.iloc[1:, 0].values
        })
    else:
        price0 = fut_df.iloc[window_start:window_end-1]['close']
        stock_precict = np.array(price0) * (1 + np.array(forecast_list))
        gt = np.array(fut_df.iloc[window_start+1:window_end]['close'])
        fut = np.array(df.iloc[window_start+1:window_end]['close'])
 
        print_feature = 'Return'
        plt_data = pd.DataFrame({
            'Date': fut_df.date[window_start+1:window_end],
            'Stock_Predict': stock_precict,
            'Stock_Truth': gt,
            'Future_Price': fut
        })
        
    
    r_squared = r2_score(plt_data['Stock_Truth'], plt_data['Stock_Predict'])
    # Normalize the data
    scaler = StandardScaler()
    plt_data[['Stock_Predict', 'Stock_Truth', 'Future_Price']] = scaler.fit_transform(plt_data[['Stock_Predict', 'Stock_Truth', 'Future_Price']])
    stock = pair['CU9999.XSGE'][idx]
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(plt_data['Date'], plt_data['Stock_Truth'], label=f'CU Truth', color='#0000FF', linewidth=2) # Standard blue
    ax.plot(plt_data['Date'], plt_data['Stock_Predict'], label=f'CU Predict', linestyle='dashed', color='#3399FF', linewidth=2) # Lighter blue, close to standard blue
    ax.plot(plt_data['Date'], plt_data['Future_Price'], label=f'{stock} Price', color='red', linewidth=2)


    plt.title(f'Prediction of CU Future using {feature} of {stock}, R2={r_squared:.2f}')
    plt.xlabel('Time')
    plt.ylabel(f'Standardized Price')
    plt.legend(title=f'Price (Standardized)')
    return fig
    
    


if __name__ == '__main__':
    pass