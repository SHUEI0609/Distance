# AI Distance Estimation System 📏

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)

単眼カメラの映像から、対象（人物など）までの距離をリアルタイムで推定するAIシステムです。
物体検出モデル **YOLOv8** と、独自にファインチューニングした **ResNet50ベースの距離推定モデル** を組み合わせることで、特定の対象を認識しながらその距離を計測します。

##  アーキテクチャ (System Architecture)

このプロジェクトは、精度と速度を両立させるために2段階のパイプライン処理を採用しています。

1.  **物体検出 (Object Detection)**
    * **Model**: `YOLOv8 Nano (yolov8n.pt)`
    * **Role**: 映像フレームから対象（Person）の位置（Bounding Box）を高速に検出します。
2.  **距離推定 (Distance Estimation)**
    * **Model**: Custom ResNet50 (`distance_model.pth`)
    * **Role**: 検出された領域の情報および画像特徴量から、カメラから対象までの物理的な距離を回帰予測します。

##  ファイル構成 (File Structure)
```
.
├── distance.py         # メイン実行スクリプト (推論パイプライン)
├── distance_model.pth  # 学習済み距離推定モデル (ResNet50ベース)
├── yolov8n.pt          # 物体検出モデル (YOLOv8)
├── requirements.txt    # 依存ライブラリ一覧
└── README.md           # ドキュメント
```
## 使い方 (Usage)
1. 環境構築
Python 3.10以上の環境で、必要なライブラリをインストールします。
```
pip install -r requirements.txt
```
2. 実行
Webカメラ、または動画ファイルを指定して実行します。
```
python distance.py
```

##  技術的なこだわり (Technical Highlights)
### ハイブリッドなモデル構成:

検出精度と処理速度のバランスを考慮し、検出には軽量な YOLOv8 Nano を採用。一方で、距離推定には表現力の高い ResNet50 を採用し、それぞれ最適なモデルを選定しました。

### 独自のデータセット作成:

距離推定モデルの学習には、実際にカメラと対象の距離を計測して作成したカスタムデータセットを使用しました。

多様な環境下でのデータを収集し、モデルのロバスト性を高める工夫を行っています。

### 転移学習 (Transfer Learning):

ImageNetで事前学習されたResNet50の重みを初期値とし、距離回帰タスク向けにファインチューニングを行うことで、少ない学習データでも高い汎化性能を実現しました。

##  今後の課題 (Future Works)
精度の向上: 現在は1m以内でしか制度が出ないが，1m以上の長距離でも制度が出るようにする.

軽量化: エッジデバイス（Raspberry Pi等）での動作に向けたモデルの量子化・軽量化。

 Author
SHUEI0609

Project Repository: https://github.com/SHUEI0609/Distance
