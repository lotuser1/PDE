# Pseudo-Label Diversity Exploitation for Few-Shot Object Detection

The official implementation of Pseudo-Label Diversity Exploitation for Few-Shot Object Detection

# Requirements

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9


## Train & Inference

### Training

#### 1. Stage 1: Training base detector.

```
python tools/train_net.py --num-gpus 1 \
        --configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml
```

#### 2. Random initialize  weights for novel classes.

```
python tools/ckpt_surgery.py \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN/model_final.pth \
        --method randinit \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN
```


#### 3. Stage 2: Fine-tune for novel data.

```
python tools/train_net.py --num-gpus 1 \
        --configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```


#### 4. Stage 3: Fine-tune for pseudo data.

```
python3 -m tools.genarate_pseudo --num-gpus 1

python3 -m tools.train_feature --num-gpus 1   
```

#### Evaluation

To evaluate the trained models, run

```
python tools/test_net.py --num-gpus 1 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
        --eval-only
```

