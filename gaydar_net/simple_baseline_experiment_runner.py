from SimpleBaselineNet import SimpleBaselineNet
from experiment_runner_base import ExperimentRunnerBase
from celeba_dataset import CelebADataset

import torch
import torch.nn as nn
from torchvision import transforms

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, batch_size, num_epochs, model_path,
                 num_data_loader_workers, lr, log_validation):

        ############ 2.3 TODO: set up transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ############

        train_dataset = CelebADataset(image_dir="data/img_align_celeba",
            annotation_txt_path="data/final_data.txt",
            data_partition_path="data/list_eval_partition.txt",
            train_val_flag="train",
            transform=transform,
        )

        val_dataset = CelebADataset(image_dir="data/img_align_celeba",
            annotation_txt_path="data/final_data.txt",
            data_partition_path="data/list_eval_partition.txt",
            train_val_flag="val",
            transform=transform,
        )

        model = SimpleBaselineNet()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, model_path, num_data_loader_workers, log_validation)

        ############ 2.5 TODO: set up optimizer
        # no need to train GoogLeNet parameters
        self.optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr = lr,
            momentum=0.9,
            #weight_decay=1e-5,
        )
        ############


    def _optimize(self, predicted_orientation, ground_truth_orientation, weight=19):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        # Binary classification, weighted loss, gay : straight = 9 : 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weight])).cuda

        loss = criterion(predicted_orientation, ground_truth_orientation)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        ############
