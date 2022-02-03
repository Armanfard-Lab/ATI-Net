import torch
from atinet import *
from padnet import *
from mtan import *
from single import *
from split import *
from utils import *
from create_dataset import *

class args():
  def __init__(self, tasks, dataroot, weight='dwa', temp=2.0, save_data_path='none', save_model_path='none', load_model_path='none'):
    self.tasks = tasks
    self.weight = weight
    self.dataroot = dataroot
    self.temp = temp
    self.save_data_path = save_data_path
    self.save_model_path = save_model_path
    self.load_model_path = load_model_path

if __name__ == "__main__":
    
    opt = args(tasks=['segmentation', 'depth', 'normal'],
           dataroot='nyuv2',
           save_data_path='ati_nyu',
           save_model_path='ati_nyu',
           load_model_path='none')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ATI_Net(tasks=opt.tasks, num_out_channels={'segmentation': 13, 'depth': 1, 'normal': 3}).to(device) # specifiy the desired model here

    if opt.load_model_path != 'none':
        print('Loading Backbone')
        pretrained_state_dict = torch.load(f'{opt.load_model_path}')
        model = load_backbone(pretrained_state_dict, model)

    optimizer = optim.Adam(model.parameters(), lr=0.5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=101, gamma=0.5)

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(model),
                                                            count_parameters(model) / 24981069))
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

    train_set = NYUv2(root=opt.dataroot, train=True)
    test_set = NYUv2(root=opt.dataroot, train=False)

    batch_size = 2
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    multi_task_trainer(train_loader,
                    test_loader,
                    model,
                    device,
                    optimizer,
                    scheduler,
                    opt,
                    100)
