#!unzip /home/workspace/input/人工智能赛道（河南）/人工智能赛道A榜数据.zip -d/home/workspace/output/rengongzhineng

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

data_root_path='/home/workspace/output/rengongzhineng/toUser'
# 读取训练数据
train_data_path = data_root_path + '/train.csv'
test_data_path = data_root_path + '/testA.csv'

# 确保路径正确，并读取数据
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data.head()

train_data.columns

train_data['sl_type'].value_counts()

def preprocess_data(df):
    # 填充缺失值，类别特征使用众数填充，数值特征使用中位数填充
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')
    
    # 获取类别和数值列
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # 不对 user_id 列进行处理
    categorical_cols = categorical_cols.drop('user_id', errors='ignore')
    
    # 填充缺失值
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # 编码 area_code 列
    if 'area_code' in df.columns:
        le_area_code = LabelEncoder()
        df['area_code'] = le_area_code.fit_transform(df['area_code'])
        
    # 编码主副卡
    if 'zfk_type' in df.columns:
        df['zfk_type'] = df['zfk_type'].map({'是': 1, '否': 0})
        
    # 提取 join_date 列中的年份
    # 提取 join_date 列中的年份并转换为整数
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date']).dt.year.astype(int)
        
    if 'jt_5gwl_flag' in df.columns:
        df['jt_5gwl_flag'] = df['jt_5gwl_flag'].map(lambda x: 1 if x == 'is_5gwl_user' else 0)
        
    # 编码 品牌 列
    if 'term_brand' in df.columns:
        le_area_code = LabelEncoder()
        df['term_brand'] = le_area_code.fit_transform(df['term_brand'].astype(str))
        
    if 'sl_type' in df.columns:
        le_area_code = LabelEncoder()
        df['sl_type'] = le_area_code.fit_transform(df['sl_type'])
    
    return df

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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
        
    # 提取 join_date 列中的年份并转换为整数
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date']).dt.year.astype(int)
        
    # 编码 jt_5gwl_flag 列
    if 'jt_5gwl_flag' in df.columns:
        df['jt_5gwl_flag'] = df['jt_5gwl_flag'].map(lambda x: 1 if x == 'is_5gwl_user' else 0)
        
    # 编码 term_brand 列
    if 'term_brand' in df.columns:
        le_term_brand = LabelEncoder()
        df['term_brand'] = le_term_brand.fit_transform(df['term_brand'].astype(str))
    
    return df

train_data = preprocess_data(train_data)
train_data

test_data= preprocess_data(test_data)
test_data

train_data['sample_flag'].value_counts()

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

train_data=feature_engineering(train_data)
test_data=feature_engineering(test_data)

# 选择指定的列
selected_columns = [
    'avg3_tc_ll','avg3_tw_ll','age','avg3_video_app1_cnt','avg3_video_app2_cnt', 'avg3_video_app_ll', 'avg3_music_app1_cnt',
    'avg3_music_app2_cnt', 'avg3_music_app_ll', 'avg3_game_app1_cnt',
    'avg3_game_app2_cnt', 'avg3_game_app_ll', 'sample_flag'
]

selected_data = train_data[selected_columns]
selected_data.head(50)

#!pip install imblearn
#!pip install --upgrade scikit-learn imbalanced-learn

import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm

# 自定义评估函数
def custom_multiclass_eval(preds, train_data, precision_weight=0.7):
    labels = train_data.get_label()
    num_class = len(np.unique(labels))
    preds = np.argmax(preds.reshape(num_class, -1), axis=0)
    
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    
    recall_weight = 1 - precision_weight
    score = (precision_macro * precision_weight + recall_macro * recall_weight) * 100
    
    return 'custom_score', score, True

# 提取特征和标签
X = train_data.drop(columns=['user_id', 'sample_flag'])
y = train_data['sample_flag']

# 将标签从1,2,3重新编码为0,1,2
y = y - 1

# 10折交叉验证
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []
test_preds = []

for i, (train_index, val_index) in enumerate(tqdm(skf.split(X, y)), start=1):
    print(f"Fold {i}: Start")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # 直接使用原始训练数据
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    valid_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    
    params = {
        'objective': 'multiclass',
        'metric': 'None',  # 省略metric，使用自定义评估函数
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_split_gain': 0.1,
        'min_child_weight': 10,
        'min_child_samples': 20,
        'max_depth': -1,
        'verbosity': -1,
        'seed': 42 , # 固定种子
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]

    model = lgb.train(params, train_data_lgb, valid_sets=[train_data_lgb, valid_data_lgb],
                      num_boost_round=200, feval=custom_multiclass_eval, callbacks=callbacks)

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_val_pred_classes = y_val_pred.argmax(axis=1)
    
    # 将预测和真实标签从0,1,2转换回1,2,3
    y_val_pred_classes = y_val_pred_classes + 1
    y_val = y_val + 1
    
    precision = precision_score(y_val, y_val_pred_classes, average='macro', zero_division=1)
    recall = recall_score(y_val, y_val_pred_classes, average='macro', zero_division=1)
    score = (precision * 0.7 + recall * 0.3) * 100
    
    scores.append(score)
    tqdm.write(f"Fold {i}: Precision={precision:.4f}, Recall={recall:.4f}, Score={score:.4f}")
    
    # 在测试集上进行预测并累加
    X_test = test_data.drop(columns=['user_id'])
    fold_test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds.append(fold_test_preds)

# 计算平均验证分数
mean_score = np.mean(scores)
print(f"Mean Validation Score: {mean_score}")

# 计算每个样本的平均预测值
mean_test_preds = np.mean(test_preds, axis=0)

# 将平均预测结果转化为类别
test_predictions_classes = mean_test_preds.argmax(axis=1)

# 将预测结果从0,1,2转换回1,2,3
test_predictions_classes = test_predictions_classes + 1

# 输出测试集的预测结果
test_predictions_classes

import os
submission = pd.DataFrame({
    'user_id': test_data['user_id'],
    'predtype': test_predictions_classes
})
submission_file_path =os.path.join(data_root_path, 'lgb1.csv')
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved to '{submission_file_path}'.")

submission['predtype'].value_counts()

submission['predtype'].value_counts()

submission

import os
# 生成提交文件的路径
submission_file_path = os.path.join(data_root_path, 'lgb1.csv')
!castlecli --third honghu --token 4c44d07ce5b52d3a85a77290b1e0a6d4 --source {submission_file_path}

