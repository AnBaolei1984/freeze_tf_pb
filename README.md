适用于将TensorFlow 1.x训练的chekpoint冻结成freeze pb用于部署。

输入参数：

1. check point的前缀
2. 输出节点的名字，以，隔开，一定要是模型真正输出节点的名字
3. 输出的frozen pb文件

示例：

python3 freeze_pb.py  uncased_L-12_H-768_A-12/bert_model.ckpt bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1 output/frozen.pb
