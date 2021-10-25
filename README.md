# RandLA-Net on SemanticPOSS
Repo at [EvernightAurora/RandLA-Net-1](https://github.com/EvernightAurora/RandLA-Net-1)

## Code
代码基于RandLA-NET的[源代码](https://github.com/QingyongHu/RandLA-Net)，git日志中记录了所作的修改。主要文件如下
- [helper_requirements.txt](helper_requirements.txt) 需求pip包
- [helper_tool.py](helper_tool.py) 多种辅助函数，包括加载点云、预处理和可视化等。
- [RandLANet.py](RandLANet.py) 网络主类，包括网络结构、训练、评估等
- [main_SemanticKITTI.py](main_SemanticKITTI.py) 主程序，为网络提供输入并驱动其运行

如要运行，除相应包外有C代码需要编译，参见[原说明文档](READIT.md)。

## Branches
为方便对代码进行管理，基础修改完成之后直接进行分支来实现不同的架构和功能。主分支为 `master` ，可视化分支为 `visualization` ，各ablation分支如下
- `ap_to_maxpool` 将AP块替换为max pooling
- `ap_to_meanpool` 将AP块替换为mean pooling
- `ap_to_sumpool` 将AP块替换为sum pooling
- `rm-locse` 删去LocSE块
- `simp_rb` 简化DRB



## Evaluate部分的说明

本代码以SemanticKITI中 00-08中所有去掉了人和骑手的帧作为样本
00-07训练集，08验证集
同时略微修改了网络，使之可以设定强制开启Dropout层，以满足第一部分的测试需要

stddef: 存储了各种全局变量
main_SemanticKITTI.py  --mode[train test] --type[0:Normal, 1:强制开启Dp, 2 3:在test时指定]


train: 如上，训练参数全部使用原来的参数， 对于剔除的标签1、2 计算会出现nan，我们不会将其计入mIoU

test: 通过加载模型文件，来跑指定的数据集。 会按照概率多次采集一个点周围的点，直到所有点的已知程度足够高
	type:1: 强制开启SD，通过--exa=k 来指定是第几次（会将结果存储到k有关的文件夹）
	type:2 or 3: 不开启SD， 但是会把包括训练集在内的所有都跑一遍（为了MD相关计算）

	type1  2 3会输出.prob和.label文件，表示预测的概率（softmax前）和标签
		prob为 shape=(T, 11), dtype=np.float32  label为 shape=(T,), dtype=np.uint32 
		次序与给定数据集相同
	type2 3会额外输出.t_label文件，为数据集标注的标签，
		上面几个数据都通过np.fromfile来读取



conf/SD_Calc 通过多次的开启Dropout, 来计算每个点在M=5次softmax前输出结果的方差var以及平均结果pred
conf/ODIN_Calc 通过对softmax前的加上T系数的类softmax，得到pred以及对应最高的类softmax概率
