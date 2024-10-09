import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# 忽略所有警告
warnings.simplefilter(action='ignore', category=Warning)

mydir = '/home/workspace/output/ai/toUser/'
train = pd.read_csv('/home/workspace/output/ai/toUser/train.csv')
testA = pd.read_csv('/home/workspace/output/ai/toUser/testA.csv')
testB = pd.read_csv('/home/workspace/output/ai/toUser/testB.csv')

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data_new(df):
    # 删除不需要的列
    drop_columns = ['avg3_sl_ll', 'sl_ll2', 'sl_flag', 'sl_type']
    #df.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    # 将年龄大于100的值替换为众数
    age_mode = df['age'][df['age'] <= 100].mode()[0]
    df.loc[df['age'] > 100, 'age'] = age_mode

    # 将gender_id大于等于2的值替换为0
    df.loc[df['gender_id'] >= 2, 'gender_id'] = 0
    
    # 获取类别和数值列
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # 不对 user_id 列进行处理
    categorical_cols = categorical_cols.drop('user_id', errors='ignore')
    
    # 填充类别特征使用众数
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # 填充数值特征使用中位数
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # 编码 area_code 列
    if 'area_code' in df.columns:
        le_area_code = LabelEncoder()
        df['area_code'] = le_area_code.fit_transform(df['area_code'])
        
    # 编码 zfk_type 列
    if 'zfk_type' in df.columns:
        df['zfk_type'] = df['zfk_type'].map({'是': 1, '否': 0})
        
    # 替换join_date中的离群值
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date'])
        threshold_date = pd.to_datetime('1995-01-01')
        third_smallest_date = df['join_date'].sort_values().unique()[2]
        df.loc[df['join_date'] < threshold_date, 'join_date'] = third_smallest_date
        
        # 将 'join_date' 转换为 datetime 类型，并确保精度到月份
        df['join_date'] = df['join_date'].dt.to_period('M')
        # 设置目标日期为2024年4月，并转换为 period 类型
        target_date = pd.Period('2024-04', freq='M')
        # 计算从 join_date 到目标日期的月份差异
        df['months_until_target'] = (target_date - df['join_date']).apply(lambda x: x.n)
        
    # 编码 jt_5gwl_flag 列
    if 'jt_5gwl_flag' in df.columns:
        df['jt_5gwl_flag'] = df['jt_5gwl_flag'].map(lambda x: 1 if x == 'is_5gwl_user' else 0)
        
    # 编码 term_brand 列
    if 'term_brand' in df.columns:
        le_term_brand = LabelEncoder()
        df['term_brand'] = le_term_brand.fit_transform(df['term_brand'].astype(str))
    
    return df

from sklearn.preprocessing import StandardScaler
def feature_engineering(data):
    
    # 衍生比值特征
    data['video_app_cnt_ratio'] = data['avg3_video_app2_cnt'] / (data['avg3_video_app1_cnt'] + 1)
    data['music_app_cnt_ratio'] = data['avg3_music_app2_cnt'] / (data['avg3_music_app1_cnt'] + 1)
    data['game_app_cnt_ratio'] = data['avg3_game_app2_cnt'] / (data['avg3_game_app1_cnt'] + 1)

    # 衍生平均值特征
    data['avg_video_app_cnt_rato1'] = data['avg3_video_app_ll'] /(data['avg3_video_app1_cnt']+1) 
    data['avg_music_app_cnt_rato1'] = data['avg3_music_app_ll'] /(data['avg3_music_app1_cnt']+1) 
    data['avg_game_app_cnt_rato1'] = data['avg3_game_app_ll'] /(data['avg3_game_app1_cnt']+1) 

    data['avg_video_app_cnt_rato2'] = data['avg3_video_app_ll'] /(data['avg3_video_app2_cnt']+1) 
    data['avg_music_app_cnt_rato2'] = data['avg3_music_app_ll'] /(data['avg3_music_app2_cnt']+1) 
    data['avg_game_app_cnt_rato2'] = data['avg3_game_app_ll'] /(data['avg3_game_app2_cnt']+1) 

    # 基于标签分布区间进行离散化
    bins = [0, 30, 40, 50, 60, 70, 1000]
    labels = [0, 1, 2, 3, 4, 5]
    data['age_bin'] = pd.cut(data['age'], bins=bins, labels=labels, include_lowest=True)
    data['age_bin'] = data['age_bin'].astype(int)  # 转换为整数类型
    
    drop_columns=['user_type','jt_5gwl_flag']
    #data=data.drop(columns=drop_columns)
    
    # 数据标准化
    scaler = StandardScaler()
    feature_columns = data.columns.difference(['user_id', 'sample_flag', 'age_bin'])  # 需要标准化的列
    #data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    return data

processed_train = preprocess_data_new(train)
processed_testA = preprocess_data_new(testA)
processed_testB = preprocess_data_new(testB)

fe_train = feature_engineering(processed_train)
fe_testA = feature_engineering(processed_testA)
fe_testB = feature_engineering(processed_testB)

X_train = fe_train.drop(['user_id', 'sample_flag', 'join_date'], axis=1)
y_train = fe_train['sample_flag']
X_testA = fe_testA.drop(['user_id', 'join_date'], axis=1)
X_testB = fe_testB.drop(['user_id', 'join_date'], axis=1)

import catboost as cb
from catboost import Pool, cv, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# 定义自定义评估指标
class CustomMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # 指示这个指标值越大越好
        return True

    def evaluate(self, approxes, target, weight):
        preds = np.argmax(approxes, axis=0)
        num_classes = len(np.unique(target))
        precision_list = []
        recall_list = []

        for cls in range(num_classes):
            true_positive = np.sum((preds == cls) & (target == cls))
            false_positive = np.sum((preds == cls) & (target != cls))
            false_negative = np.sum((preds != cls) & (target == cls))

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

        # 计算宏观平均
        macro_precision = np.mean(precision_list)
        macro_recall = np.mean(recall_list)

        custom_score = (macro_precision * 0.7 + macro_recall * 0.3) * 100
        return custom_score, 1


# *******超参数调参*********

# 创建训练数据池
data_pool = Pool(data=X_train, label=y_train)

# 定义搜索空间
space = {
'iterations': hp.quniform('iterations', 100, 500, 50),
    'depth': hp.quniform('depth', 4, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'grow_policy': hp.choice('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    'bootstrap_type': hp.choice('bootstrap_type', ['Bernoulli', 'No']),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'random_strength': hp.uniform('random_strength', 0, 10),
    'rsm': hp.uniform('rsm', 0.6, 1.0),
    'loss_function': 'MultiClass',
    'eval_metric': CustomMetric(), # 自定义评估指标
    'use_best_model': True,
    'od_type': "Iter",
    'boosting_type': hp.choice('boosting_type', ['Ordered', 'Plain']), 
    'random_seed': 42
}

# 定义目标函数
def objective(params):
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    
    # 检查 grow_policy 并调整 boosting_type
    if params['grow_policy'] != 'SymmetricTree' and params['boosting_type'] == 'Ordered':
        params['boosting_type'] = 'Plain'
    if params['bootstrap_type'] == 'No': 
        params.pop('subsample', None) # 移除 subsample 参数
    
    cv_results = cv(
        pool=data_pool,
        params=params,
        fold_count=5,  # 5折交叉验证
        shuffle=True,
        partition_random_seed=42,
        verbose=False,
    )
    
    # 由于自定义评估指标是越大越好，因此我们取负数作为损失
    best_score = np.max(cv_results['test-CustomMetric-mean'])
    
    return {'loss': -best_score, 'status': STATUS_OK}
    
# 进行超参数优化
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

# 整理超参数
best_params['iterations'] = int(best_params['iterations']) 
best_params['depth'] = int(best_params['depth'])

# 需要将返回的整数映射回对应的字符串值
best_params['grow_policy'] = ['SymmetricTree', 'Depthwise', 'Lossguide'][best_params['grow_policy']]
best_params['bootstrap_type'] = ['Bernoulli', 'No'][best_params['bootstrap_type']]
best_params['boosting_type'] = 'Plain' if best_params['boosting_type'] == 1 else 'Ordered'

# 检查 bootstrap_type 并处理 subsample
if best_params['bootstrap_type'] == 'No':
    best_params.pop('subsample', None)
    
# 确保 grow_policy 和 boosting_type 兼容
if best_params['grow_policy'] in ['Depthwise', 'Lossguide'] and best_params['boosting_type'] == 'Ordered':
    best_params['boosting_type'] = 'Plain'

# 打印最佳参数
print("Best parameters:", best_params)

# 记录最佳参数
params = {
    'iterations': 200,           
    'depth': 9,
    'learning_rate': 0.08302766255577063,
    'l2_leaf_reg': 4.673906541373009,
    'rsm': 0.8694581718022971
    'random_strength': 3.2439950378069717,
    'boosting_type': 'Plain',
    'bootstrap_type': 'No',
    'grow_policy': 'Depthwise',
    'loss_function': 'MultiClass',
    'eval_metric': CustomMetric(),  # 使用自定义评估指标
    # 'use_best_model': True,
    'od_type': "Iter",
    'random_seed': 42
}

# 使用最佳参数进行训练
model = CatBoostClassifier(**params)

# 训练模型
model.fit(data_pool)

# 在测试集上进行预测
y_pred_b = model.predict(X_testB)

# 保存预测结果到 CSV 文件
submit_b_example = pd.read_csv('/home/workspace/input/人工智能赛道（河南）/submitB.csv')
submit_b_example['predtype'] = y_pred_b
submit_b_example.to_csv(mydir + 'result/team/result_cat_0924_3.csv', index=False) 
