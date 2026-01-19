import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ==========================================
# 設定
# ==========================================
MODEL_PATH = 'distance_model.pth'  # 添付されたモデルファイル名
MAX_DISTANCE = 10.0                # 学習時と同じ正規化パラメータ(10m)

# ★CPUを強制的に指定
DEVICE = torch.device("cpu")

# ==========================================
# 1. モデル定義 (学習時の構造と合わせる)
# ==========================================
class DistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50のバックボーン
        # weights=None: 推論時は学習済み重みをロードするので初期値は何でも良い
        backbone = models.resnet50(weights=None) 
        
        # 特徴抽出部
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        num_ftrs = backbone.fc.in_features
        
        # 回帰ヘッド (MLP + Sigmoid)
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ==========================================
# 2. メイン処理
# ==========================================
def main():
    print(f"動作モード: {DEVICE} (GPUなし)")

    # --- A. 距離推定モデルのロード ---
    print("距離モデルを読み込んでいます...")
    try:
        dist_model = DistanceModel().to(DEVICE)
        
        # ★重要: GPUで保存したモデルをCPUにマッピングしてロード
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        dist_model.load_state_dict(state_dict)
        
        dist_model.eval() # 推論モードに固定
        print("✅ 距離モデル読み込み完了")
    except FileNotFoundError:
        print(f"エラー: {MODEL_PATH} が見つかりません。同じフォルダに置いてください。")
        return
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        print("学習時のモデル定義と推論時の定義が一致していない可能性があります。")
        return

    # 前処理定義
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- B. YOLOモデルのロード ---
    print("YOLOモデルを読み込んでいます...")
    # 初回は自動ダウンロードされます (yolov8n.pt は軽量でCPU向け)
    yolo_model = YOLO('yolov8n.pt') 
    print("✅ YOLO読み込み完了")

    # --- C. Webカメラ起動 ---
    cap = cv2.VideoCapture(0) # 0番のカメラ
    if not cap.isOpened():
        print("エラー: カメラが見つかりません。")
        return

    print("開始します。終了するには画像ウィンドウ上で 'q' キーを押してください。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOで物体検出 (CPUで実行されます)
        results = yolo_model(frame, verbose=False)
        
        # 検出結果の処理
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 信頼度が低いものはスキップ (0.5未満は無視)
                if float(box.conf[0]) < 0.5:
                    continue

                # 座標取得
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # クラス名取得
                cls_id = int(box.cls[0])
                label_name = yolo_model.names[cls_id]

                # --- 画像切り抜き ---
                h, w, _ = frame.shape
                # 範囲外エラー防止
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0: continue

                # --- 距離推定 (CPU) ---
                # OpenCV(BGR) -> PIL(RGB)
                pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                
                # 前処理 (バッチ次元を追加: [1, 3, 224, 224])
                input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = dist_model(input_tensor)
                    # 0~1の出力をメートルに戻す
                    distance_m = output.item() * MAX_DISTANCE

                # --- 描画 ---
                # 人間は赤、それ以外は緑
                color = (0, 0, 255) if label_name == 'person' else (0, 255, 0)

                # 枠線
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # テキスト (物体名: 距離)
                text = f"{label_name}: {distance_m:.2f}m"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 画面表示
        cv2.imshow('Distance Estimation (CPU)', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()