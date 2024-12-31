# Depth Anything V2 real-time Webcam Streaming
Run Depth Anything Model V2 on a **live video stream**


[Depth Anything V2 깃헙 링크 ](https://github.com/DepthAnything/Depth-Anything-V2)


## Demo

![damv2](https://github.com/user-attachments/assets/3d8df92a-5baf-41ae-8518-610d3e9cb62b)


</div>

## Pre-trained Models

모델을 다운받구 checkpoints 폴더에 pth 모델을 넣어줍니다.

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |


또는 아래 명령어로 모델 다운로드 

```bash
cd checkpoints

./download_ckpts.sh

cd ..
```

## Usage

### 가상환경 준비 

```bash 
conda create -n dam2 python=3.11 -y

conda activate dam2

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```

### 필요 라이브러리 설치 

```bash 
pip install -r requirements.txt
```

### real-time Webcam Depth-Anything V2

> 00.run_webcam_local.py