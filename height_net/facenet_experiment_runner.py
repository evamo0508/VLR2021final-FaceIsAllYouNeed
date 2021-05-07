from facenet import FaceNet
from experiment_runner_base import ExperimentRunnerBase
from celeba_dataset import CelebADataset

import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.insert(0, "../")
from utils.augmentation import *

class FaceNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, batch_size, num_epochs, model_path,
                 num_data_loader_workers, lr, log_validation, pretrained='vggface2'):
        
                
        assert pretrained in ['vggface2', 'casia-webface'], \
        "pretrained should be in ['vggface2', 'casia-webface']"

        ############ 2.3 TODO: set up transform
        train_tf = train_aug_with_random_crop(224)
        val_tf = val_aug(224)
        ############

        train_dataset = CelebADataset(image_dir="../data/img_align_celeba",
            annotation_txt_path="../data/final_data.txt",
            data_partition_path="../data/list_eval_partition.txt",
            train_val_flag="train",
            transform=train_tf,
        )

        val_dataset = CelebADataset(image_dir="../data/img_align_celeba",
            annotation_txt_path="../data/final_data.txt",
            data_partition_path="../data/list_eval_partition.txt",
            train_val_flag="val",
            transform=val_tf,
        )

        model = FaceNet(pretrained=pretrained)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, model_path, num_data_loader_workers, log_validation)

        ############ 2.5 TODO: set up optimizer
        # no need to train GoogLeNet parameters
        self.optimizer = torch.optim.SGD(
            filter(lambda p:p.requires_grad, self._model.parameters()),
            lr = lr,
            momentum=0.9,
            #weight_decay=1e-5,
        )
        ############


    def _optimize(self, predicted_height, ground_truth_height):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        criterion = nn.MSELoss(reduction='mean').cuda()
        loss = criterion(predicted_height, ground_truth_height)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        ############
