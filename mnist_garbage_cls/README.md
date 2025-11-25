# Introduction

本项目基于EfficientNet-B0构建了一个26分类的垃圾分类模型。



项目结构说明：

`train_main.py`文件存放了模型训练的代码。

`predict.py`文件存放了MO平台的接口测试文件。在MO平台提交时，按照指引新建一个cell，将`predict.py`文件的内容拷贝到新的cell中即可提交。

`src`文件夹中存放了EfficientNet-B0的Backbone预训练权重。该文件仅在训练时会使用。

`results`文件夹中存放了训练好的模型权重。在这里，我放置了两个权重文件，分别为`best_model.pth`和`best_model2.pth`，你可以自行选择一个进行提交。我测试下来是`best_model2.pth`权重文件较优。

> 由于我并没有对模型进行细致“炼丹”，因此最终测试的分数仅为96.15。如果你对分数有较高要求，可以使用`train_main.py`文件对超参数进行调优以获得性能更好的模型。