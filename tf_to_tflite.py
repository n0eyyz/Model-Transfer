import tensorflow as tf

# 1. SavedModel 경로 (폴더명!)
saved_model_dir = "clipvision_tf"  # 네 SavedModel 폴더명 그대로!

# 2. 변환기 생성
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 3. 변환 (옵션 커스터마이즈 가능!)
tflite_model = converter.convert()

# 4. 결과 파일로 저장
with open("clipvision.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite 변환 및 저장 완료!")
