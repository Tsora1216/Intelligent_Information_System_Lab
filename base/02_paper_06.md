---
title: 知能情報システム研究室　活動記録
tags: データサイエンティスト
author: Tsora1216
slide: false
---
この実験は、選択的転移強化学習において転移先の強化学習済み知識が存在する状態を想定した実験である。

目的は、前回のSAP-netのライブラリ化を行い、転移先のWebotsプログラムに移植できるSAP-netの作成を行うことである。

# ライブラリの作成
下記のようにライブラリを作成していく
## initの作成
```
from TakayaSora.SAPnet import *
```

## LICENSEの作成
```
MIT License

Copyright (c) 2023 TakayaSora

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## READMEの作成
```
# TakayaSora

使用できる関数は下記のとおりです。

## データベースのセットアップ
SQL_SetUp("database.sqlite")

## データの取得と表示
df = SQL_GetData("database.sqlite")
print(df)

## 角度と距離を入力
angle = float(input('角度を入力してください: '))
distance = float(input('距離を入力してください: '))

## SAP-netにベクトル情報を渡して拡散
SAP_df = SAP_net(df,angle,distance)

## SAP-netが選んだ知識を出力
select_knowledge = selection(SAP_df)
print(SAP_df)
print(select_knowledge)
```

## Setupの作成
```
from setuptools import setup, find_packages

setup(
    name='TakayaSora',
    version='0.1',
    packages=find_packages()
)
```

上記のファイルでライブラリを作成していく

# ライブラリ化
下記のコマンドでライブラリをインストールしていく。
```cmd
#!pip install git+https://github.com/Tsora1216/TakayaSora
```

ライブラリを使用したプログラムは下記のとおりである。

import TakayaSoraを使用する事で、コードが部品化されており、圧倒的に短く、わかりやすくなっていることが分かる。

```Python
import TakayaSora as Sora

# データベースのセットアップ
Sora.SQL_SetUp("database.sqlite")

# データの取得と表示
df = Sora.SQL_GetData("database.sqlite")
print(df)

#角度と距離を入力
angle = float(input('角度を入力してください: '))
distance = float(input('距離を入力してください: '))

#SAP-netにベクトル情報を渡して拡散
SAP_df = Sora.SAP_net(df,angle,distance)

#SAP-netが選んだ知識を出力
select_knowledge = Sora.selection(SAP_df)
print(SAP_df)
print(select_knowledge)
```

![](https://gyazo.com/3fd301989bfb28c9cc9da42422d75581.png)