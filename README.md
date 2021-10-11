# 2021 APTOS Big Data Competition

https://tianchi.aliyun.com/competition/entrance/531929/information?lang=zh-cn

请在 ./data 目录下建立这两个文件夹,

- `test-set-img`: 放测试集图片
- `training-set-img`: 放训练集图片

## 请直接向本项目 commit, 不用 fork 再 Pull requests

## 数据处理的过程

#### 拍平图片文件夹:

`flat-img-dir.py` 把原来具有文件夹层次的图片目录拍平了, 所有的图片直接放在 `data/training-set-img/` 或 `data/test-set-img/` 里.


#### 训练集:

1. `clean-training-set.py`: 清洗 `raw-data/TrainingAnnotation.csv`, 得到 `data/training-set.csv`
2. `generate-training-set-with-img.py`: 从 `data/training-set.csv`, 加上对图片的索引, 得到 `training-set-with-img.json`


#### 待提交答案的输入, 官方将它误称作测试集, 我们也继续叫测试集算了:

1. `clean-test-set.py`: 由 `raw-data/PreliminaryValidationSet_Info.csv`, 得到 `data/test-set.csv`
   - 注: 无需清洗, 只是改变了 gender 的意义(改为 0 表示 男, 1 表示女)
2. `generate-test-set-with-img.py`: 从 `data/test-set.csv`, 加上对图片的索引, 得到 `training-set-with-img.json`

## 后续训练应使用 `data/training-set-with-img.json`

它长这样:

```json
[
    {
        "patient ID": "0000-1315L",
        "gender": 1,
        "age": 76,
        "diagnosis": 6,
        "preVA": 0.3,
        "anti-VEGF": 0,
        "preCST": 227,
        "preIRF": 0,
        "preSRF": 0,
        "prePED": 0,
        "preHRF": 0,
        "VA": 0.4,
        "continue injection": 0,
        "CST": 237,
        "IRF": 0,
        "SRF": 0,
        "PED": 0,
        "HRF": 0,
        "img_before": [
            "./data/training-set-img/0000-1315L_1000.jpg",
            "./data/training-set-img/0000-1315L_1001.jpg",
            "./data/training-set-img/0000-1315L_1002.jpg",
            "./data/training-set-img/0000-1315L_1003.jpg",
            "./data/training-set-img/0000-1315L_1004.jpg",
            "./data/training-set-img/0000-1315L_1005.jpg"
        ],
        "img_after": [
            "./data/training-set-img/0000-1315L_2000.jpg",
            "./data/training-set-img/0000-1315L_2001.jpg",
            "./data/training-set-img/0000-1315L_2002.jpg",
            "./data/training-set-img/0000-1315L_2003.jpg",
            "./data/training-set-img/0000-1315L_2004.jpg",
            "./data/training-set-img/0000-1315L_2005.jpg"
        ]
    },
    {
        "patient ID": "0000-0656L",
        "gender": 0,
        "age": 64,
        "diagnosis": 2,
        "preVA": 0.32,
        "anti-VEGF": 1,
        "preCST": 201,
        "preIRF": 0,
        "preSRF": 1,
        "prePED": 1,
        "preHRF": 0,
        "VA": 0.4,
        "continue injection": 1,
        "CST": 191,
        "IRF": 0,
        "SRF": 0,
        "PED": 1,
        "HRF": 0,
        "img_before": [
            "./data/training-set-img/0000-0656L_1000.jpg",
            "./data/training-set-img/0000-0656L_1001.jpg",
            "./data/training-set-img/0000-0656L_1002.jpg",
            "./data/training-set-img/0000-0656L_1003.jpg",
            "./data/training-set-img/0000-0656L_1004.jpg",
            "./data/training-set-img/0000-0656L_1005.jpg"
        ],
        "img_after": [
            "./data/training-set-img/0000-0656L_2000.jpg",
            "./data/training-set-img/0000-0656L_2001.jpg",
            "./data/training-set-img/0000-0656L_2002.jpg",
            "./data/training-set-img/0000-0656L_2003.jpg",
            "./data/training-set-img/0000-0656L_2004.jpg",
            "./data/training-set-img/0000-0656L_2005.jpg"
        ]
    },
    ......
}
```