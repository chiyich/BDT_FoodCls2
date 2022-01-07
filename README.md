# Food_cls src files for forme's team.

Src files for SIGS_Big_Data_ML_Exam_2021.

https://www.kaggle.com/t/b7ed697207f0401b94a1f5c49c559d68

# Member
（按姓名首字母排序）
池逸尘 段毅甫 王思琦 叶雨欣

# Environment
- python 3.8
- torch 1.9.0
- torchvision 0.10.0
- tqdm

必须用 `GPU` 跑 Q.Q
- NVIDIA GeForce GTX1080 Ti

# Download

下载数据到指定路径 `./data/food/`,将三个文件夹分别移动到:

- `./data/food/train`
- `./data/food/val`
- `./data/food/test`

# Prepare

生成索引文件，创建数据集：

`python prepare.py --src /data/food/train --out /data/food/train.txt`

`python prepare.py --src /data/food/val  --out /data/food/val.txt`

修改 `dataset.py` 的 `107-108` 行为你的指定路径

# Hyper-parameter

修改 `config.py` 的超参数为你需要的值

`root`修改为你的项目本地路径

# Train

`export CUDA_VISIBLE_DEVICES=X`

`python train.py`

# Inferance

功能为用训练好的best模型测试 `test` 路径下的所有图片，并生成 `result.txt` 文件

 `python inferance.py` 

# Models and Results

现在有几个模型：ResNet34、DenseNet121

0.462->0.485->0.506

