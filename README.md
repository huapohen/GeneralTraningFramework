# General Framework
 
 ## Pretrained
 code: https://pan.baidu.com/s/1cnlodDkddKAjCJY0OZ-KgQ?pwd=gpsd 
 efficientformer_v2 | Accuracy=96.78%  
 mobilenet_v2 | Accuracy=83.45%  
 dataset: https://pan.baidu.com/s/1pW-FZezkO8xoPO9aeQ6XTA?pwd=gpsd 

 
 ## Train
启动训练，有三种方式：  
  1. python train.py  
  2. python train.py  -- your-params.json地址
  3. python search_hyperparameters.py

## Test
启动测试，有两种方式：
  1. python evaluate.py  
  2. python evaluate.py  -- your-params.json地址

## Debug
  1. 调试不可用search_hyperparamters.py起，用train.py和evaluate.py
  2. vscode 和 ipdb.set_trace()，注意文件相互引用和当前执行路径
  3. dataloader debug时，需把num_workers=0和prefetch_factor不设置

## Description
  1. 方式1启动时，用指定params启动训练或测试时，需要在params.json里指定所有需要用到的参数，  
例如数据集地址：experiments/params.json-->data_dir
  2. 方式2启动时，需要在parameters.py里指定所有需要设置的参数，它是去覆盖（和增加）已有的默认experiments/params.json参数。
  3. 方式3启动时，是通过search_pyperparamters.py调用train.py，它至少需要传入 gpu_used/model_dir/exp_name/tb_path和exp_id参数

## Path Setting
  1. 参数"data_dir","exp_root_dir"和"exp_current_dir"均需指定，否则用默认参数
  2. exp_name需指定，否则报错
  3. experiments里对应实验的文件夹和存params.json的config文件夹，采用软连接的方式，实际路径不在项目代码里，方便。

## Distributed Training
  1. 需完善
