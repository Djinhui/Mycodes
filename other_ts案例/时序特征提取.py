import pandas as pd

data = pd.read_csv('Train.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')

'''
Date-time Based Features:

year、month、day、hour、minute、second、microsecond、date、time、timetz
dayofyear、weekofyear、week、dayofweek、weekday、weekday_name、quarter、days_in_month
is_month/year/quarter_start/end、is_leap_year
'''

# ==============Date-Related Features==============
data['year'] = data['Datetime'].dt.year
data['month'] = data['Datetime'].dt.month
data['day'] = data['Datetime'].dt.day
data['dayofweek_num'] = data['Datetime'].dt.dayofweek
data['dayofweek_name'] = data['Datetime'].dt.weekday_name


# ==============Time-Based Features================
data['Hour'] = data['Datetime'].dt.hour
data['minute'] = data['Datetime'].dt.minute


# ===============Lag-Features=================
# The lag value we choose will depend on the correlation of individual values with its past values.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data['Count'], lags=10)
plot_pacf(data['Count'], lags=10)

data['lag_1'] = data['Count'].shift(1)
data['lag_2'] = data['Count'].shift(2)



# =====================Grouoby Feature============================
def code_mean(df, cat_feature, num_feature):
    return df.groupby(cat_feature)[num_feature].mean().to_dict()
data['month_mean'] = list(map(code_mean(data, 'month', 'y').get, data.month))

# ======================Rolling Window Feature=====================
# window-size固定为7，窗口向前移动
data['rolling_mean'] = data['Count'].rolling(window=7).mean()
data['rolling_max'] = data['Count'].rolling(window=7).max()
data['rolling_min'] = data['Count'].rolling(window=7).min()
data['rolling_std'] = data['Count'].rolling(window=7).std()


# ======================Expanding Window Feature===============
# with every step window-size increases by one as it takes into account every new value in the series
data['expanding_mean'] = data['Count'].expanding(2).mean()


# ====================Domain-Specific Features==================


