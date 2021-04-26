### gan对抗生成人脸(DCGAN_FACE_GENERATION)

===========================
#### 00-项目信息
```
作者：TuringEmmy
时间:2021-4-26 16:36:37
详情：目前在LFW数据集上可以跑通，由于没有合适的硬件，没法查看结果，运行结果效果非常差
gan相关算法之前没有接触过，项目开发较为缓慢，边学边做
```
#### 01-环境依赖
```
ubuntu18.04
python3.7
tensorflow1.14
```
#### 02-部署步骤
##### 训练
```
sh scripts/data.sh
sh scripts/server.sh
sh scripts/train.sh
```
#### 03-目录结构描述
```
.
│  data.py
│  dcgan_face_generation.py
│  models.py
│  README.md
│  server.py
│  utils.py
│
└─scripts
        data.sh
        server.sh
        train.sh
```


#### 04-版本更新
##### V1.0.0 版本内容更新-2021-4-26 17:08:03
- LSW的数据集预处理，模型构建，模型训练，模型预测都完成


#### 05-TUDO
- 修改成tensorflow2.x版本的
- 针对celeba也做处理