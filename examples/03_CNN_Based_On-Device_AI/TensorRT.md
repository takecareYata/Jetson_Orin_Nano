# Tensor RT 동작 필요 명령어

PATH 설정 (한번만)

* echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc

* source ~/.bashrc

04번 예제 동작

* cd examples/03_CNN_Based_On-Device_AI/

* trtexec --onnx=RPS_MobileNetV2.onnx --saveEngine=RPS_MobileNetV2.engine

* python EX_04_Board_RPS_MobileNetV2.py



