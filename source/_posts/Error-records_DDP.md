---
title:  Error-records:DDP
author: pljj315
comment: gitalk
date: 2025-03-17 13:42:30
tags:
- DDP分布式训练
category: Error records
index_img: ../imgs/Research-Insights_Stylized-generation-based-diffusion/image-20250227153104659.png
---





1. Linear实现MLP ：输入数据形式：形状为[N, *, in_features]的tensor，N为batch size，这个参数是PyTorch各个数据操作中都具备的，相似的，输出数据形式为[N, *, out_features]
   -  n_features：每个输入样本的大小，对应MLP中当前层的输入节点数/特征维度

   - out_features：每个输出样本的大小，对应MLP中当前层的输出节点数/特征维度

2. 验证del 如何删？set_ipadapter的原因？

3. 9cinpainting训练ipa——minus_clip

4. 8c 训练ipa——minus_clip


------------
## 分布式训练过程报错记录：

### 1. 多卡训练发生NCCL错误，单卡不会发生：

- 更新分布式训练相关库acelerate deepspeed xformers transformers；

- 进程通信方式设置：

  `依然使用P2P通信： export NCCL_P2P_LEVEL=NVL `

  `直接取消P2P通信：export NCCL_IB_DISABLE=1 && export NCCL_P2P_DISABLE=1`

- dataloader中设置 `pin_memory=False`;  loss取消`gather`操作并且loss的打印和写入log时要`.detach().item()`，

- `accelerator.sync_gradients, accelerator.is_local_main_process, accelerator.is_main_process`的使用和检查

- log_validation中谨慎：

  - 在log_validation后没多少个step就NCCL卡主，time.sleep()取代log_validation操作后能够正常训练：

    说明log_validation内有操作破坏了正常训练中的某步骤，经检查发现： log_validation中的**set_ipadapter()**设置问题导致attn_processor被重新设置，丢失权重、梯度？


### 2. log_validation推理出全黑图：

- pipe的输入不符合要求或者其他问题导致pipe的推理过程**出错**，但是在分布式训练过程中不一定会显示报错，所以需要单独把log_validation拿出来过一遍
- 上述步骤依然没报错，可能是VAE的模型权重版本问题，比如inpaintPipeline的时候使用stabilityai/stable-diffusion-xl-base-1.0的VAE会不匹配，出图全黑！还是得使用"madebyollin/sdxl-vae-fp16-fix"

### 3. overflow问题：

fp16改为bf16，代码中数据的.to(weight_dtype)也要修改

### 4. out of memory（OOM）: 

- 数据精度：fp32改为bf16/fp16

- 优化器：使用use_8bit_adamw

- 训练数据批次大小：调小batchsize/num of worker/ accumulate_step

- 增加为多卡训练：有时模型以及优化器状态以及梯度的总体参数量较大，需要更多卡
- deepspeed的zero_stage: 1 改为2，改为3

### 5. huggingface load 模型：

已经设置过环境变量`export HF_HOME=".../huggingface"`，此目录下也已经保存了模型，但是load不进来:` from_pretrained(, local_files_only = True)`

### 6. 过拟合same device:

换了小数据集进行过拟合实验时，报错：NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000，发现报错位置是dataloader的prepare,检查bs ·num_processes· gradient_accumulation_steps > 总训练数据大小，也就是可能只用了一张卡就load了全部的数据，改小bs· num_processes· gradient_accumulation_steps并未解决， 降低了accelerate==0.25.0版本解决；



## ssh 报错记录：

### 1. vscode忽然连接不上远程服务器，一直显示Opening Remote

- 进入远程主机的个人目录，删除.vscode-server，路径一般是/home/usrname/.vscode-server
- 删除本地：known_hosts和known_hosts.old，路径一般是C:\Users$(usrname).ssh
- 原文链接：https://blog.csdn.net/qq_36183881/article/details/123873537



### wandb

1. 当Ctrl+C无法立即结束时, 通过`pkillwandb-service`命令快速停止wandb-service的输出
2. [BrokenPipeError on Ubuntu machine](https://community.wandb.ai/t/brokenpipeerror-on-ubuntu-machine/4117)  proxy或者网络中断导致：本地机器与wandb服务器通讯中断，导致wandb日志记录线程中断，导致训练中断！！！太不智能了！还是使用**tensorboard**吧
