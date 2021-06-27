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