# Version Check
# TensorFlow : 2.10.0
# Keras : 2.10.0
# onnx-tf : 1.10.0
# onnx : 1.12.0
# tensorflow-probability : 0.16.0
# Python Version : 3.8 or 3.9
# conda create -n (env name) python=3.8

from onnx_tf.backend import prepare
import onnx

# 1. 변환할 ONNX 파일명
onnx_model = onnx.load("clip_vision.onnx")

# 2. TF로 변환
tf_rep = prepare(onnx_model)

# 3. SavedModel로 내보내기
tf_rep.export_graph("clipvision_tf")

print("변환 완료! 'clipvision_tf' 폴더가 생성됐으면 성공!")