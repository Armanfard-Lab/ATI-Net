# ATI-Net: Attentive Task Interaction Network for Multi-Task Learning


PyTorch implementation of the paper: "Attentive Task Interaction Network for Multi-Task Learning"

<center><img src="https://github.com/Armanfard-Lab/ATI-Net/network_fig.png" alt="Overview" width="800" align="center"></center>

## Citation

You can find the preprint of our paper on arXiv. (link "arXiv" with the following url: https://arxiv.org/abs/2201.10649)

## Abstract

Multitask learning (MTL) has recently gained a lot of popularity as a learning paradigm that can lead to improved per-task performance while also using fewer per-task model parameters compared to single task learning. One of the biggest challenges regarding MTL networks involves how to share features across tasks. To address this challenge, we propose the Attentive Task Interaction Network (ATI-Net). ATI-Net employs knowledge distillation of the latent features for each task, then combines the feature maps to provide improved contextualized information to the decoder. This novel approach to introducing knowledge distillation into an attention based multitask network outperforms state of the art MTL baselines such as the standalone MTAN and PAD-Net, with roughly the same number of model parameters.

## Usage

In the main.py:
	- Configure the dataroot, train/test data saving path, model parameter saving path, and model loading path (if you want to load a backbone)
	- Specify the model you would like to train
	- Set the hyperparameters (optimizer, scheduler, batch size, epochs)
	- Run main.py

The preprocessed NYUv2 dataset can be found here. (link "here" with the following url: "https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0")

NOTE: To save time during experimentation, we would load a half-trained MTAN backbone when training ATI-Net, and then proceed with the second half of the training. This is effectively equivalent to the training stretegy specified in the paper. However, this allows us to experiment with ATI-Net much faster since we don't need to restart training every experiment.

## References

Our codebase builds upon and borrows elements from the public implimentation of MTAN (link "MTAN" with the following url: https://github.com/lorenmt/mtan). Particularly, the trainer, dataloaders, and baseline models are taken from there. We also borrow elements for the distillation modules from the following MTL repo. (link "MTL repo" with the following url: https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)
