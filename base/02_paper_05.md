---
title: 知能情報システム研究室　活動記録
tags: データサイエンティスト
author: Tsora1216
slide: false
---
この実験は、選択的転移強化学習において転移先の強化学習済み知識が存在する状態を想定した実験である。

目的は、前回のSAP-netの高速化を行い、転移先が用意されればすぐに転移できるSAP-netの作成を行うことである。

# プログラムをまとめる
今まで表示していた、グラフをすべて消去し、実験4までのプログラムをすべて格納
```python
# 新しいデータを追加
new_id = len(df) + 1
new_activation = -1.0
new_description = '障害物'
new_angle = float(input('角度を入力してください: '))
new_distance = float(input('距離を入力してください: '))

new_data = pd.DataFrame({
    'id': [new_id],
    'activation': [new_activation],
    'description': [new_description],
    'angle': [new_angle],
    'distance': [new_distance]
})

input_df = pd.concat([df, new_data], ignore_index=True)

# idとdescriptionを結合した文字列を作成
input_df['id_description'] = input_df['id'].astype(str) + '_' + input_df['description']

# ベクトルとして角度と距離を使用するため、データを準備
vectors = input_df[['angle', 'distance']]

# ベクトル間のユークリッド距離を計算
distances = np.linalg.norm(vectors.values[:, np.newaxis] - vectors.values, axis=2)

# クロス表に距離を格納
cross_table = pd.DataFrame(distances, index=input_df['id_description'], columns=input_df['id_description'])

# ユークリッド距離を求めたcross_table
#print(cross_table)

# ユークリッド距離の評価指標を計算し、再度DataFrameに格納
max_distance = np.nanmax(cross_table.values)  # ユークリッド距離の最大値（NaNを除く）
evaluated_values = 1 - cross_table.values / max_distance
activation_table = pd.DataFrame(evaluated_values, index=cross_table.index, columns=cross_table.columns)

# ユークリッド距離を正規化した値をdfに格納
#print(activation_table)

# 評価指標を1/10にスケーリング
activation_table_div10 = activation_table / 10
activation_table_min1 = 1-activation_table

# 再度activation_tableを表示
print(activation_table_div10)
print(activation_table_min1)

input_df2=input_df.copy()

print(input_df)
print(input_df2)
```
![](https://gyazo.com/553e86af7e5e63cd94ea00f341a3e6b7.png)

上記のプログラムで処理を準備・前処理以外の処理を一つにまとめた。

次に、準備や前処理も一つのプログラムにまとめ、全体の処理速度を見る。

```
# 新しいデータを追加
new_id = len(df) + 1
new_activation = -1.0
new_description = '障害物'
new_angle = float(input('角度を入力してください: '))
new_distance = float(input('距離を入力してください: '))

new_data = pd.DataFrame({
    'id': [new_id],
    'activation': [new_activation],
    'description': [new_description],
    'angle': [new_angle],
    'distance': [new_distance]
})

input_df = pd.concat([df, new_data], ignore_index=True)

# idとdescriptionを結合した文字列を作成
input_df['id_description'] = input_df['id'].astype(str) + '_' + input_df['description']

# ベクトルとして角度と距離を使用するため、データを準備
vectors = input_df[['angle', 'distance']]

# ベクトル間のユークリッド距離を計算
distances = np.linalg.norm(vectors.values[:, np.newaxis] - vectors.values, axis=2)

# クロス表に距離を格納
cross_table = pd.DataFrame(distances, index=input_df['id_description'], columns=input_df['id_description'])

# ユークリッド距離を求めたcross_table
#print(cross_table)

# ユークリッド距離の評価指標を計算し、再度DataFrameに格納
max_distance = np.nanmax(cross_table.values)  # ユークリッド距離の最大値（NaNを除く）
evaluated_values = 1 - cross_table.values / max_distance
activation_table = pd.DataFrame(evaluated_values, index=cross_table.index, columns=cross_table.columns)

# ユークリッド距離を正規化した値をdfに格納
#print(activation_table)

# 評価指標を1/10にスケーリング
activation_table_div10 = activation_table / 10
activation_table_min1 = 1-activation_table

input_df2=input_df.copy()

for i in range(len(activation_table_div10.columns)-1):
    activity_value_temp = activation_table_div10.loc[activation_table_div10.columns[i], activation_table_div10.columns[-1]]
    input_df2.loc[input_df2['id_description'] == activation_table_div10.columns[i], 'activation'] += activity_value_temp

# 画像を格納するリスト
images = []

while not (input_df2['activation'] > 1).any():
    for i in range(len(activation_table_div10.columns)):
        for j in range(len(activation_table_div10.columns)):
            if i==j:
                continue
            activity_value_temp = activation_table_div10.loc[activation_table_div10.columns[i], activation_table_div10.columns[j]]
            input_df2.loc[input_df2['id_description'] == activation_table_div10.columns[i], 'activation'] += activity_value_temp

# descriptionの最大値を持つレコードを出力
max_description = input_df2['activation'].max()
max_records = input_df2[input_df2['activation'] == max_description]
print(max_records['id_description'][0])
```

その結果、33.6秒の処理を0.3秒に短縮することができた。
112倍の速度に加速させることができたと言える。

![](https://gyazo.com/362009717251c47be209f0b155d2fb7f.png)


次に可視化機能を捨て、高速化したプログラムを関数化していく。

~SAP関数~
```Python
import os
import io
import glob
import base64
import sqlite3
import datetime
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from PIL import Image
import matplotlib as mpl
import japanize_matplotlib
import matplotlib.pyplot as plt
from IPython import display as dd
from matplotlib.font_manager import FontProperties


def SQL_SetUp(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # データベースファイルが存在する場合は処理を終了
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = cursor.fetchall()
    if existing_tables:
        print("Database already exists. Exiting setup.")
        conn.close()
        return

    # テーブルを作成
    cursor.execute('''
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            angle FLOAT,
            distance FLOAT,
            activation FLOAT DEFAULT 0,
            description TEXT
        )
    ''')

    data = [
        ("直進", 333.44, 1.12, 0),#１番の知識
        ("左寄りの直進", 345.97, 1.03, 0),#２番の知識
        ("右寄りの直進", 0, 1, 0),#１番の知識
        ("右寄りの直進", 14.04, 1.03, 0),#１番の知識
        ("直進", 26.56, 1.12, 0)#１番の知識
    ]

    # データを挿入
    cursor.executemany("INSERT INTO knowledge (description, angle, distance, activation) VALUES (?, ?, ?, ?)", data)

    conn.commit()
    conn.close()

def SQL_GetData(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # データを取得してDataFrameに格納
    cursor.execute("SELECT * FROM knowledge")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

    conn.close()

    return df

# 新しいデータを追加
def SAP_net(df,new_angle,new_distance):
    new_id = len(df) + 1
    new_activation = -1.0
    new_description = '障害物'

    new_data = pd.DataFrame({
        'id': [new_id],
        'activation': [new_activation],
        'description': [new_description],
        'angle': [new_angle],
        'distance': [new_distance]
    })

    input_df = pd.concat([df, new_data], ignore_index=True)

    # idとdescriptionを結合した文字列を作成
    input_df['id_description'] = input_df['id'].astype(str) + '_' + input_df['description']

    # ベクトルとして角度と距離を使用するため、データを準備
    vectors = input_df[['angle', 'distance']]

    # ベクトル間のユークリッド距離を計算
    distances = np.linalg.norm(vectors.values[:, np.newaxis] - vectors.values, axis=2)

    # クロス表に距離を格納
    cross_table = pd.DataFrame(distances, index=input_df['id_description'], columns=input_df['id_description'])

    # ユークリッド距離を求めたcross_table
    #print(cross_table)

    # ユークリッド距離の評価指標を計算し、再度DataFrameに格納
    max_distance = np.nanmax(cross_table.values)  # ユークリッド距離の最大値（NaNを除く）
    evaluated_values = 1 - cross_table.values / max_distance
    activation_table = pd.DataFrame(evaluated_values, index=cross_table.index, columns=cross_table.columns)

    # ユークリッド距離を正規化した値をdfに格納
    #print(activation_table)

    # 評価指標を1/10にスケーリング
    activation_table_div10 = activation_table / 10
    activation_table_min1 = 1-activation_table

    input_df2=input_df.copy()

    for i in range(len(activation_table_div10.columns)-1):
        activity_value_temp = activation_table_div10.loc[activation_table_div10.columns[i], activation_table_div10.columns[-1]]
        input_df2.loc[input_df2['id_description'] == activation_table_div10.columns[i], 'activation'] += activity_value_temp

    # 画像を格納するリスト
    images = []

    while not (input_df2['activation'] > 1).any():
        for i in range(len(activation_table_div10.columns)):
            for j in range(len(activation_table_div10.columns)):
                if i==j:
                    continue
                activity_value_temp = activation_table_div10.loc[activation_table_div10.columns[i], activation_table_div10.columns[j]]
                input_df2.loc[input_df2['id_description'] == activation_table_div10.columns[i], 'activation'] += activity_value_temp

    return input_df2

def selection(input_df2):
    # descriptionの最大値を持つレコードを出力

    return select_knowledge
```

~Mainプログラム~
```python
# データベースのセットアップ
SQL_SetUp("database.sqlite")

# データの取得と表示
df = SQL_GetData("database.sqlite")
print(df)

#角度と距離を入力
angle = float(input('角度を入力してください: '))
distance = float(input('距離を入力してください: '))

#SAP-netにベクトル情報を渡して拡散
SAP_df = SAP_net(df,angle,distance)

#SAP-netが選んだ知識を出力
select_knowledge = selection(SAP_df)
print(SAP_df)
print(select_knowledge)
```

上記のプログラムでSAP-netの入力から出力までを集約し、一つのプログラムにすることができた。