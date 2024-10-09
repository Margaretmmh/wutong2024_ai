# wutong2024_ai
China Mobile "Wutong" Competiton AI Track

# 数字内容推荐模型 - README

## 项目背景

随着人民生活水平的不断提高，人们在追求物质生活的同时，对丰富精神生活的需求也日益旺盛。我们坚持“以人民为中心”的理念，牢牢把握“数字经济”新赛道带来的重要机遇，锚定“世界一流信息服务科技创新公司”的发展定位，致力于推动业务从传统的通信服务向信息服务扩展。作为国内领先的通信服务提供商，我们不仅拥有庞大的用户群体，还具备丰富的自有数字内容产品。然而，在面对海量数字内容的情况下，如何精准地为用户推荐他们感兴趣的产品，提升用户体验和满意度，增加信息服务收入，成为我们亟需解决的关键挑战。

为了实现这一目标，我们决定运用大数据技术和机器学习算法，建立一个高效的数字内容推荐模型。该模型旨在通过对用户特征、业务订购、业务使用、套餐资费、内容洞察等多维度数据进行深入挖掘，建立精确的用户画像，并以此为基础为用户推荐符合其个性化需求的数字内容产品，促成用户订购相关权益和内容产品，从而提升公司信息服务收入。

## 项目概述

本项目旨在通过构建一个数字内容推荐模型，从用户的各种特征数据中挖掘其对数字内容的潜在兴趣。我们使用了一系列的数据预处理、特征工程技术，尝试了单独使用LightGBM模型、单独使用CatBoost模型以及模型融合三种方式，最终采用了基于LightGBM的多分类模型，最终实现了对用户类型的准确预测，为数字内容产品营销提供了可靠的潜客数据支持。

## 数据预处理

在数据预处理环节，我们主要进行了以下操作：

1. **缺失值处理**：
   - 类别特征：采用众数填充。
   - 数值特征：采用中位数填充，确保数据的完整性和一致性。

2. **异常值处理**：
   - `age`字段：将大于100的异常年龄值替换为该字段的众数。
   - `gender_id`字段：将大于等于2的值替换为0，保持性别编码的一致性。

3. **类别编码**：
   - 使用`LabelEncoder`对类别特征如`area_code`, `term_brand`等进行编码。
   - `zfk_type`字段被映射为二进制值，'是'映射为1，'否'映射为0。
   - `join_date`字段提取年份并转换为整数，表示用户加入的时间信息。
   - `jt_5gwl_flag`字段也被转换为二进制值，标识用户是否为5G网络用户。

4. **数据标准化**：
   - 使用`StandardScaler`对数值特征进行了标准化处理，以提升模型的训练速度和效果。

## 特征工程

特征工程是模型成功的关键步骤，我们在此过程中设计了多种特征，以提高模型的预测能力：

1. **衍生比值特征**：
   - 计算不同APP使用次数的比率，如`video_app_cnt_ratio`，`music_app_cnt_ratio`等，捕捉用户在不同数字内容领域的使用偏好。

2. **衍生平均值特征**：
   - 针对APP使用次数与流量之间的比率，生成平均值特征，如`avg_video_app_cnt_ratio1`，以反映用户的内容使用深度。

3. **交互特征**：
   - 计算APP总使用次数及各类APP在总使用中的占比，如`video_app_usage_ratio`，以此评估用户的内容偏好和互动强度。

4. **时序特征**：
   - 通过对比不同时间段的使用数据，计算APP使用趋势特征，如`video_app_usage_trend`，用于捕捉用户行为的变化趋势。

5. **统计特征**：
   - 针对APP使用次数，计算其均值、最大值、最小值和标准差，生成统计特征，以全面反映用户的使用习惯。

6. **离散化特征**：
   - 将年龄分为不同的区间（如30岁以下，30-40岁等），并进行离散化处理，以便更好地分析不同年龄段用户的行为特征。

## 模型构建与训练

在模型构建与训练阶段，我们尝试了单独使用LightGBM模型、单独使用CatBoost模型以及模型融合三种方式，最终采用了LightGBM模型进行用户类型的多分类预测。具体步骤如下：

1. **数据准备**：
   - 从预处理后的数据集中提取特征和标签，去除`user_id`等无关特征。
   - 将标签进行重新编码，以便于模型的多分类训练。

2. **模型训练**：
   - 使用5折交叉验证（StratifiedKFold）对模型进行训练和验证，以避免模型的过拟合问题。
   - 在每一折中，利用自定义评估函数，结合Precision和Recall两个指标，对模型进行综合评估。

3. **模型参数设置**：
   - 通过调整超参数，如`num_leaves`、`learning_rate`、`feature_fraction`等，来优化模型的性能。
   - CatBoost模型使用`hyperopt`结合5折交叉验证进行超参数调参

4. **模型评估**：
   - 在每一折验证中，计算Precision、Recall和自定义得分，并取5折的平均值作为最终模型的性能指标。

5. **模型预测**：
   - 在测试集上进行预测，将交叉验证的预测结果取平均值作为最终预测结果，并生成提交文件。

## 模型评估

模型评估使用了自定义的评分函数，该函数结合了Precision和Recall两个指标，确保模型不仅能够准确预测用户类型，还能最大限度地覆盖所有潜在用户，提升数字内容产品的订购成功率。

在5折交叉验证中，我们得到了稳定且较高的得分，表明模型在实际应用中具有较强的泛化能力。

## 环境配置说明

本项目基于Python编程语言开发，主要使用了以下库：

- `pandas`：用于数据处理和分析。
- `numpy`：用于数值计算和矩阵操作。
- `scikit-learn`：用于数据预处理、模型评估等。
- `lightgbm`：用于模型训练与预测。
- `tqdm`：用于显示训练过程中的进度条。
- `catboost`：用于模型训练与预测。
- `hyperopt`：用于超参数调参。

要在本地环境中运行本项目，请确保安装上述依赖库，可以通过以下命令快速安装：

```bash
pip install pandas numpy scikit-learn lightgbm tqdm catboost hyperopt
```

## 总结

通过本项目，我们成功地构建了一个高效的数字内容推荐模型，该模型能够根据用户的特征数据，精准地预测用户对不同数字内容的兴趣，并推荐最符合其需求的产品。这不仅提升了用户的使用体验，也为公司增加了信息服务收入提供了有力支持。

本模型在复杂的数据预处理、精细的特征工程以及强大的机器学习算法支持下，展示出了出色的性能和良好的应用前景。未来，我们可以进一步优化模型，探索更多的特征组合和算法创新，以持续提升推荐效果。
