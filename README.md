这是一个语音克隆项目

一定需要在gpu环境上运行

git clone

python pipline.py
首先会下载vctk数据集 
这个需要从hugging face 11G左右 需要换个镜像 你懂得

之后会下载Ecapa-TDNN模型

之后看看tqdm时间就行，还有看看logging 文件，loss爆了或者nan 停下就行
每100个epoch会保存模型

有报错再说吧