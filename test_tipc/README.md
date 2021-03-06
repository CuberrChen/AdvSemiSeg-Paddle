
# 从训练到推理部署工具链测试方法介绍

test_train_inference_python.sh和params.txt文件配合使用，完成分割模型从训练到预测的流程测试。

# 安装依赖
- 安装PaddlePaddle >= 2.1.2
- 安装PaddleSeg依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

# 目录介绍

```bash
test_tipc/
├── configs/advsemiseg_deeplabv2_res101_humanseg/train_infer_python.txt                                  # 测试分割模型的参数配置文件
                        └── advsemiseg_deeplabv2_res101_humanseg_192x192_mini_supervisely.yml   # 测试分割模型的config配置文件
└── prepare.sh                                                    # 完成test_model.sh运行所需要的数据和模型下载
└── test_train_inference_python.sh                                                 # 测试主程序
```

# 使用方法

test_train_inference_python.sh包含四种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```bash
bash test_tipc/prepare.sh ./test_tipc/configs/advsemiseg_deeplabv2_res101_humanseg/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/advsemiseg_deeplabv2_res101_humanseg/train_infer_python.txt 'lite_train_lite_infer'
```


# 日志输出
最终在```test_tipc/output```目录下生成.log后缀的日志文件
