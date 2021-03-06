import os
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for our celebA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, model_path, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._log_freq = 100   # Steps
        self._test_freq = 250  # Steps
        self._save_model_freq = 1000 # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers, drop_last=True) # drop last during train to prevent batchnorm break

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        # for logging
        self._model_path     = model_path
        self._log_validation = log_validation
        self._height_mean    = train_dataset._height_mean
        self._height_std     = train_dataset._height_std
        print("height mean {}, std {}".format(self._height_mean, self._height_std))

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, iteration):
        ############ 2.8 TODO
        # Should return your validation accuracy
        loss, err = 0.0, 0.0
        with torch.no_grad():
            num_batches = len(self._val_dataset_loader)

            for batch_id, batch_data in enumerate(self._val_dataset_loader):
                batch_img = batch_data["image"].cuda()
                batch_height = batch_data["height"].squeeze().cuda()

                predicted_height = self._model(batch_img) # (N, )
                original_height = batch_height.detach().cpu() * self._height_std + self._height_mean

                diff = (predicted_height - batch_height).detach().cpu()
                loss += torch.sum(diff ** 2)
                err += torch.sum(torch.abs(diff) * self._height_std / original_height)

                # for visualization
                if batch_id == 0:
                    vis_idx = 0
                    # input img
                    inv_normalize = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                    )
                    log_img = inv_normalize(batch_data["image"][vis_idx])
                    # predicted height (denormalized)
                    log_pred = predicted_height[vis_idx] * self._height_std + self._height_mean
                    # GT height
                    log_gt = batch_height[vis_idx] * self._height_std + self._height_mean

        MSE = loss / (num_batches * self._batch_size)
        ACC = (1 - err / (num_batches * self._batch_size)) * 100
        ############

        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            self.tb.add_image("Image", log_img, iteration)
            self.tb.add_scalar("Prediction", log_pred, iteration)
            self.tb.add_scalar("Ground Truth", log_gt, iteration)
            ############
        return MSE, ACC

    def train(self):

        # init tensorboad visualization
        self.tb = SummaryWriter()

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your ptimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                batch_img = batch_data["image"].cuda()
                batch_height = batch_data["height"].squeeze().cuda()

                predicted_height = self._model(batch_img)
                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_height, batch_height)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss.item()))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    if self._log_validation:
                        self.tb.add_scalar("Training loss", loss.item(), current_step / self._log_freq)
                    ############

                #if batch_id == num_batches - 1:
                if batch_id == num_batches - 1 or epoch + batch_id == 0:
                    self._model.eval()
                    val_loss, val_acc = self.validate(epoch)
                    print("Epoch: {} has val loss {}, acc {}%".format(epoch, val_loss, val_acc))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    if self._log_validation:
                        self.tb.add_scalar("Validation acc", val_loss, epoch)
                    ############

                # save models
                if current_step % self._save_model_freq == 0:
                    model_saving_path = os.path.join("models", self._model_path)
                    if not os.path.exists(model_saving_path):
                        os.makedirs(model_saving_path)
                    torch.save(self._model.state_dict(), os.path.join(model_saving_path, "epoch" + str(epoch) + "step" + str(current_step) + ".pth"))

        self.tb.close()
