import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import statistics
from random import gauss
from statistics import mean
import random
import seaborn as sns
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

# 四捨五入関数
def round_half_up(n):
    return float(Decimal(str(n)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

# データ分布可視化関数
def plot_distribution_overlap(df1, df2, use_model_name, alpha=0.5):
    # 各データフレームの列名を取得
    columns = df1.columns

    # 各列に対してプロット
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df1[col], color='blue', kde=True, stat="density", linewidth=0, alpha=alpha, label='Generated Data')
        sns.histplot(df2[col], color='red', kde=True, stat="density", linewidth=0, alpha=alpha, label='Original Data')
        plt.title(f'Distribution of {col} - {use_model_name}', fontsize=18)
        if 'distance' in col:
            plt.xlabel(f'{col} [m]', fontsize=18)  # x軸ラベルに単位を追加
            plt.xlim(0, 100000)  # x軸の表示範囲を0から50000に設定
        else:
            plt.xlabel(col, fontsize=18)

        # plt.xlabel(col, fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.legend()
        plt.show()
        
def white_noise(df, test_df, excluded_col, round_half_up_col, zero_col, target_name, batch_size, model):

    use_model_name = 'WhiteNoise'

    df_user = df.drop(excluded_col, axis=1)

    # 数値列に対してループを回し、ホワイトノイズを加える
    for column in df_user.columns:
        if df_user[column].dtype == 'float64' or df_user[column].dtype == 'int64':
            # 元のデータの分散に基づいてホワイトノイズを生成
            # random.seed(42)
            # np.random.seed(42)
            whitenoise = [gauss(mu = 0.0, sigma = statistics.pstdev(df_user[column])) for i in range(df_user.shape[0])]
            df_user[column] += whitenoise

    
    # Replace negative values with 0
    # 負の値を0に置き換え
    combined_df[zero_col] = df_user[zero_col].applymap(lambda x: max(x, 0))

    # Rounding half up
    if round_half_up_col:
        for col in round_half_up_col:
            combined_df[col] = combined_df[col].apply(round_half_up)

    # Data Distribution Visualization
    # データ分布可視化
    plot_distribution_overlap(combined_df, df_user, use_model_name)

    print(f'Generated data size:{combined_df.shape}')
    print(f'Original data size:{df_user.shape}')

    gen_df = pd.concat([combined_df, df_user])
    test = test_df.drop(excluded_col, axis=1)
    test = test.drop(target_name, axis=1)

    # List to store prediction results
    # 予測結果を格納するリスト
    predictions = []

    X_train = gen_df.drop(target_name, axis=1)
    y_train = gen_df[target_name]

    X_test = test.drop(target_name, axis=1)
    y_test = test[target_name]

    # model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test
