import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import UnetPlusPlus
import torch.nn.functional as F

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer


class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """

    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
    ):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model

        """
        super().__init__()
        self.save_hyperparameters()

        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(
                in_channels,
                out_channels,
                model_params["depth"],
                model_params["kernel_size"],
                model_params["pool_sizes"],
            )
        elif model_type == "UNet":
            self.model = UNet(
                in_channels,
                out_channels,
                model_params["n_encoders"],
                model_params["embedding_size"],
                model_params["scale_factor"],
            )
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(
                in_channels, out_channels, model_params["scale_factor"]
            )
        elif model_type == "UNetxx":
            self.model = UnetPlusPlus(
                encoder_name= model_params['encoder_name'], encoder_depth=model_params['depth'], encoder_weights=model_params['encoder_weights'], decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, in_channels=1, classes=4, activation=None, aux_params=None)
        else:
            raise ValueError(f"model_type {model_type} not recognized")


        # define performance metrics for segmentation task
        # such as accuracy per class accuracy, average IoU, average F1 score, average AUC, etc.
        # these metrics will be logged to weights and biases        
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=4)
            
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=4)
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=4)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=4)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=4)
        
        self.train_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=4)
        self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=4)

    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        if isinstance(self.model, UnetPlusPlus):
            # padding is required to prevent input channel errors
            _, _, height, width = X.shape
            pad_height = (32 - height % 32) % 32
            pad_width = (32 - width % 32) % 32
            
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            if pad_height > 0 or pad_width > 0:
                X = F.pad(X, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        X = self.model.forward(X)
        if isinstance(self.model, UnetPlusPlus):
            if X.size(2) > 200 and X.size(3) > 200:
                padding_removed = (X.size(2) - 200) // 2
                X = X[:, :, padding_removed:padding_removed+200, padding_removed:padding_removed+200]
            downscale = nn.MaxPool2d(50)
            return downscale(X)
        return X

    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height),
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        # assuming data needs to be casted to float32
        casted_img = batch[0].to(torch.float32)

        y = self.forward(casted_img)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(
            y, batch[1].squeeze(1).to(torch.int64)
        )  # squeeze and cast the mask to int64

        # log metrics
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(y, batch[1].squeeze(1)))
        self.log("train_auc_avg", self.train_auc(y, batch[1].squeeze(1).to(torch.int32)))
        self.log("train_f1_avg", self.train_f1(y, batch[1].squeeze(1)))
        self.log("train_iou_avg", self.train_iou(y, batch[1].squeeze(1)))

        # class 0
        y_pred_0 = y[:, 0, :, :]
        y_true_0 = batch[1].squeeze(1) == 0
        self.log("train_f1_0", self.train_f1(y_pred_0, y_true_0))

        # class 1
        y_pred_1 = y[:, 1, :, :]
        y_true_1 = batch[1].squeeze(1) == 1
        self.log("train_f1_1", self.train_f1(y_pred_1, y_true_1))

        # class 2
        y_pred_2 = y[:, 2, :, :]
        y_true_2 = batch[1].squeeze(1) == 2
        self.log("train_f1_2", self.train_f1(y_pred_2, y_true_2))
        
        # class 3
        y_pred_3 = y[:, 3, :, :]
        y_true_3 = batch[1].squeeze(1) == 3
        self.log("train_f1_3", self.train_f1(y_pred_3, y_true_3))

        return loss
        



    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """

        # batch is passed in, cast as needed
        casted_data = batch[0].to(torch.float32)

        # call self.forward
        y = self.forward(casted_data)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(y, batch[1].squeeze(1).to(torch.int64))
        
        # log metrics
        self.log("val_loss", loss)
        self.log("val_auc_avg", self.val_auc(y, batch[1].squeeze(1).to(torch.int32)))
        self.log("val_accuracy", self.val_accuracy(y, batch[1].squeeze(1)))
        self.log("val_f1_avg", self.val_f1(y, batch[1].squeeze(1)))
        
        # class 0
        y_pred_0 = y[:, 0, :, :]
        y_true_0 = batch[1].squeeze(1) == 0
        self.log("val_f1_0", self.val_f1(y_pred_0, y_true_0))

        # class 1
        y_pred_1 = y[:, 1, :, :]
        y_true_1 = batch[1].squeeze(1) == 1
        self.log("val_f1_1", self.val_f1(y_pred_1, y_true_1))

        # class 2
        y_pred_2 = y[:, 2, :, :]
        y_true_2 = batch[1].squeeze(1) == 2
        self.log("val_f1_2", self.val_f1(y_pred_2, y_true_2))
        
        # class 3
        y_pred_3 = y[:, 3, :, :]
        y_true_3 = batch[1].squeeze(1) == 3
        self.log("val_f1_3", self.val_f1(y_pred_3, y_true_3))

        return loss

    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        optimizer = Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer
