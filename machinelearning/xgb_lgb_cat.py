import pandas as pd
from sklearn.model_selection import train_test_split

flights = pd.read_csv('flights.csv')
flights = flights.sample(frac=0.01, random_state=10)
flights = flights[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT", "ORIGIN_AIRPORT","AIR_TIME","DEPARTURE_TIME", "DISTANCE", "ARRIVAL_DELAY"]]

# 对标签进行离散化，延误10分钟以上才算延误
flights["ARRIVAL_DELAY"] = (flights["ARRIVAL_DELAY"]>10)*1

cat_cols = ["AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]
# 类别特征编码
for item in cat_cols:
      flights[item] = flights[item].astype("category").cat.codes +1

X_train, X_test, y_train, y_test = train_test_split(flights.drop(["ARRIVAL_DELAY"], axis=1),
                                                    flights["ARRIVAL_DELAY"], random_state=10, test_size=0.3)


# XGB
import xgboost as xgb
from sklearn.metrics import roc_auc_score

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',   
    'gamma': 0.1,
    'max_depth': 8,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'eta': 0.001,
    'seed': 1000,
    'nthread': 4,
}
dtrain = xgb.DMatrix(X_train, y_train) # should one-hot
num_rounds = 500
model_xgb = xgb.train(params, dtrain, num_rounds)

dtest = xgb.DMatrix(X_test)
y_pred = model_xgb.predict(dtest)

print('AUC of test set based on XGB:', roc_auc_score(y_test, y_pred))

# LGB
import lightgbm as lgb
params = {
"max_depth": 5, 
"learning_rate" : 0.05, 
"num_leaves": 500,  
"n_estimators": 300
}

dtrain = lgb.Dataset(X_train, label=y_train)
cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]
model_lgb = lgb.train(params, dtrain, categorical_feature = cate_features_name)
y_pred = model_lgb.predict(X_test)
print('AUC of testset based on XGBoost: ', roc_auc_score(y_test, y_pred))

# Cat
import catboost as cb
cat_features_index = [0,1,2,3,4,5,6]

# cb.Pool(X, y)

model_cb = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=50, depth=6, iterations=300, l2_leaf_reg=1,learning_rate=0.1)
model_cb.fit(X_train, y_train, cat_features=cat_features_index)
y_pred = model_cb.predict(X_test)
print('AUC of testset based on CatBoost: ',roc_auc_score(y_test, y_pred))


# GridSearch
from sklearn.model_selection import GridSearchCV
model = xgb.XGBClassifier()
param_list = {'max_depth':[3,4,5],'n_estimators':[100, 200,300]
              }
grid_search = GridSearchCV(model, param_grid=param_list, cv=3, verbose=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

# RandomSearch
from sklearn.model_selection import RandomizedSearchCV


# Bayes Optimization


import xgboost as xgb
from bayes_opt import BayesianOptimization

# 定义目标优化函数
def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):
    # 指定要优化的超参数
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    # 定义xgb交叉验证结果
    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=5,
                   seed=random_state,
                   callbacks=[xgb.callback.early_stop(50)])
    return cv_result['test-auc-mean'].values[-1]

# 定义相关参数
num_rounds = 3000
random_state = 2021
num_iter = 25
init_points = 5
params = {
    'eta': 0.1,
    'silent': 1,
    'eval_metric': 'auc',
    'verbose_eval': True,
    'seed': random_state
}
# 创建贝叶斯优化实例
# 并设定参数搜索范围
xgbBO = BayesianOptimization(xgb_evaluate, 
                             {'min_child_weight': (1, 20),
                               'colsample_bytree': (0.1, 1),
                               'max_depth': (5, 15),
                               'subsample': (0.5, 1),
                               'gamma': (0, 10),
                               'alpha': (0, 10),
                                })
# 执行调优过程
xgbBO.maximize(init_points=init_points, n_iter=num_iter)