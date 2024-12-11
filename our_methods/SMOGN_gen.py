import smogn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
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

def SMOGN_gen(df, test_df, excluded_col, round_half_up_col, zero_col, target_name, rel_coef, model):
    use_model_name = 'SMOGN'
    smogn_dict_small = {}

    df_user = df.drop(excluded_col, axis=1)

    # 実行
    smogn_df = smogn.smoter(
        data = df_user,
        y = target_name,
        rel_coef = rel_coef
    )

    smogn_df = smogn_df.reset_index(drop=True)

    if len(df_user)>=len(smogn_df):
        pass
    else:
        sub = len(smogn_df) - len(df_user)
        smogn_df = smogn_df.drop(smogn_df.sample(n=sub).index)

    # Replace negative values with 0
    # 負の値を0に置き換え
    combined_df[zero_col] = smogn_df[zero_col].applymap(lambda x: max(x, 0))

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
