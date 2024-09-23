# 最適化アルゴリズムプロジェクト

このプロジェクトは、勾配法を可視化するためのもので、さまざまな最適化アルゴリズムを実装し、Rosenbrock関数を最適化します。以下のアルゴリズムが含まれています。

## 実装されている最適化アルゴリズム

1. **通常の勾配法 (NormalOptimizer)**
2. **モーメンタム法 (MomentumOptimizer)**
3. **ネステロフの勾配加速法 (NesterovOptimizer)**
4. **AdaGrad (AdaGradOptimizer)**
5. **RMSprop (RMSpropOptimizer)**
6. **AdaDelta (AdaDeltaOptimizer)**
7. **Adam (AdamOptimizer)**
8. **メトロポリス法 (MetropolisOptimizer)**
9. **ニュートン・ラフソン法 (NewtonRaphsonOptimizer)**

## プロジェクト構成

- `optimizer.h`: 最適化アルゴリズムのクラスと関数の宣言
- `optimizer.cpp`: 最適化アルゴリズムの実装
- `gradient_descent.cpp`: 勾配降下法の実行と結果の保存

## 使用方法

1. プロジェクトをクローンまたはダウンロードします。
2. `make`コマンドを使用してプログラムをコンパイルします。
3. 実行時に、使用する最適化手法を選択します。
4. 結果は`gradient_descent.dat`に保存され、ヒートマップデータは`heatmap.dat`に保存されます。

## 依存関係

- C++11以上のコンパイラ
- 標準ライブラリ

## 注意事項

- 各最適化アルゴリズムのパラメータは、必要に応じて調整できます。
- プロジェクトは教育目的であり、商用利用は推奨されません。

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。