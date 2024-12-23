import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import precision, recall, f1_score, retrieval_average_precision, auroc
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics.functional import precision

from torchmetrics import AUROC, Accuracy

import torchmetrics.functional as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import wandb
import warnings
import os
import numpy as np
warnings.filterwarnings('ignore')
from q2l_labeller.data.cutmix import CutMixCriterion
from q2l_labeller.loss_modules.simple_asymmetric_loss import AsymmetricLoss
from q2l_labeller.loss_modules.TwoWayMultiLabelLoss import TwoWayLoss
from q2l_labeller.models.query2label import Query2Label
from q2l_labeller.data.coco_cat import COCOCategorizer as cc    
from q2l_labeller.models.cls_cvt.cls_cvt import build_CvT
# class Query2LabelTrainModule(pl.LightningModule):
#     def __init__(
#         self,
#         data,
#         backbone_desc,
#         conv_out_dim,
#         hidden_dim,
#         num_encoders,
#         num_decoders,
#         num_heads,
#         batch_size,
#         image_dim,
#         learning_rate,
#         momentum,
#         weight_decay,
#         n_classes,
#         thresh=0.5,
#         use_cutmix=False,
#         use_pos_encoding=False,
#         loss="BCE",
#     ):
#         super().__init__()

#         # Key parameters
#         self.save_hyperparameters(ignore=["model", "data"])
#         self.data = data
#         self.auroc = AUROC(num_classes=n_classes)
#         self.model = Query2Label(
#             model=backbone_desc,
#             conv_out=conv_out_dim,
#             num_classes=n_classes,
#             hidden_dim=hidden_dim,
#             nheads=num_heads,
#             encoder_layers=num_encoders,
#             decoder_layers=num_decoders,
#             use_pos_encoding=use_pos_encoding,
#         )
#         if loss == "BCE":
#             self.base_criterion = nn.BCEWithLogitsLoss()
#         elif loss == "ASL":
#             self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
#         elif loss == "mll":
#             self.base_criterion = TwoWayLoss()

#         self.criterion = CutMixCriterion(self.base_criterion)

#     def forward(self, x):
#         x = self.model(x)
#         return x

#     def evaluate(self, batch, stage=None):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.base_criterion(y_hat, y.type(torch.float))

#         rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))

#         category_prec = precision(
#             y_hat,
#             y.type(torch.int),
#             average="macro",
#             num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh,
#             multiclass=False,
#             # task='multiclass',    
#             # num_classes = 290,
#         )
#         category_recall = tf.recall(
#             y_hat,
#             y.type(torch.int),
#             average="macro",
#             num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh,
#             multiclass=False,
#         )
#         category_f1 = tf.f1_score(
#             y_hat,
#             y.type(torch.int),
#             average="macro",
#             num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh,
#             multiclass=False,
#         )

#         overall_prec = precision(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh,multiclass=False
#         )
#         overall_recall = tf.recall(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
#         )
#         overall_f1 = tf.f1_score(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False
#         )

#         if stage:
#             self.log(f"{stage}_loss", loss, prog_bar=True)
#             self.log(f"{stage}_rmap", rmap, prog_bar=True, on_step=False, on_epoch=True)

#             self.log(f"{stage}_cat_prec", category_prec, prog_bar=True)
#             self.log(f"{stage}_cat_recall", category_recall, prog_bar=True)
#             self.log(f"{stage}_cat_f1", category_f1, prog_bar=True)

#             self.log(f"{stage}_ovr_prec", overall_prec, prog_bar=True)
#             self.log(f"{stage}_ovr_recall", overall_recall, prog_bar=True)
#             self.log(f"{stage}_ovr_f1", overall_f1, prog_bar=True)

#             # # log prediction examples to wandb
            
#             # pred = self.model(x)
#             # pred_keys = pred[0].sigmoid().tolist()
#             # pred_keys = [0 if p < self.hparams.thresh else 1 for p in pred_keys]


#             # mapper = cc()
#             # pred_lbl = mapper.get_labels(pred_keys)
            
#             # try:
#             #     self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_lbl)]})
#             # except AttributeError:
#             #     pass
        


#     def training_step(self, batch, batch_idx):
#         if self.hparams.use_cutmix:
#             x, y = batch
#             y_hat = self(x)
#             # y1, y2, lam = y
#             loss = self.criterion(y_hat, y)

#         else:
#             x, y = batch
#             y_hat = self(x)
#             loss = self.base_criterion(y_hat, y.type(torch.float))
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.evaluate(batch, "val")
#         # Calculate and log MAP@20
#         x, y = batch
#         y_hat = self(x)

#         map_at_20 = self.compute_map_at_k(y_hat.sigmoid(), y, k=20)
        
#         # self.log('val_auroc', self.auroc(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log("val_map_at_20", map_at_20, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         self.evaluate(batch, "test")

#         # Calculate and log MAP@20
#         x, y = batch
#         y_hat = self(x)
#         map_at_20 = self.compute_map_at_k(y_hat.sigmoid(), y, k=20)
#         self.log("test_map_at_20", map_at_20, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.learning_rate,
#             betas=(0.9, 0.999),
#             weight_decay=self.hparams.weight_decay,
#         )

#         lr_scheduler_dict = {
#             "scheduler": OneCycleLR(
#                 optimizer,
#                 self.hparams.learning_rate,
#                 epochs=self.trainer.max_epochs,
#                 steps_per_epoch=len(self.data.train_dataloader()),
#                 anneal_strategy="linear",
#             ),
#             "interval": "step",
#         }
#         return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
#         # return optimizer
#     def extract_features(self, dataloader):
#         self.eval()  # Set model to evaluation mode
#         features = []
#         with torch.no_grad():
#             for batch in dataloader:
#                 images, _ = batch
#                 feats = self(images, return_features=True)  # Extract features
#                 features.append(feats.detach().cpu())
#         features = torch.cat(features, dim=0)
#         return features
# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import torchmetrics.functional as tf
# from torch.optim.lr_scheduler import OneCycleLR
# from torchmetrics import AUROC
# from q2l_labeller.data.cutmix import CutMixCriterion
# from q2l_labeller.loss_modules.simple_asymmetric_loss import AsymmetricLoss
# from q2l_labeller.loss_modules.TTwoWayMultiLabelLoss import TwoWayLoss
# from q2l_labeller.models.query2label import Query2Label
# import warnings
# warnings.filterwarnings('ignore')
# import wandb

# class Query2LabelTrainModule(pl.LightningModule):
#     def __init__(
#         self,
#         data,
#         backbone_desc,
#         conv_out_dim,
#         hidden_dim,
#         num_encoders,
#         num_decoders,
#         num_heads,
#         batch_size,
#         image_dim,
#         learning_rate,
#         momentum,
#         weight_decay,
#         n_classes,
#         thresh=0.5,
#         use_cutmix=False,
#         use_pos_encoding=False,
#         loss="BCE",
#     ):
#         super().__init__()

#         self.save_hyperparameters(ignore=["model", "data"])
#         self.data = data
#         self.auroc = AUROC(num_classes=n_classes)
#         self.model = Query2Label(
#             model=backbone_desc,
#             conv_out=conv_out_dim,
#             num_classes=n_classes,
#             hidden_dim=hidden_dim,
#             nheads=num_heads,
#             encoder_layers=num_encoders,
#             decoder_layers=num_decoders,
#             use_pos_encoding=use_pos_encoding,
#         )
#         if loss == "BCE":
#             self.base_criterion = nn.BCEWithLogitsLoss()
#         elif loss == "ASL":
#             self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
#         elif loss == "mll":
#             self.base_criterion = TwoWayLoss()

#         self.criterion = CutMixCriterion(self.base_criterion)

#     def forward(self, x, return_attention=False):
#         return self.model(x, return_attention)

#     def evaluate(self, batch, stage=None):
#         x, y = batch
#         y_hat, attentions = self(x, return_attention=True)
#         loss = self.base_criterion(y_hat, y.type(torch.float))

#         rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))
#         category_prec = tf.precision(
#             y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh, multiclass=False)
#         category_recall = tf.recall(
#             y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh, multiclass=False)
#         category_f1 = tf.f1_score(
#             y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
#             threshold=self.hparams.thresh, multiclass=False)

#         overall_prec = tf.precision(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)
#         overall_recall = tf.recall(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)
#         overall_f1 = tf.f1_score(
#             y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)

#         if stage:
#             self.log(f"{stage}_loss", loss, prog_bar=True)
#             self.log(f"{stage}_rmap", rmap, prog_bar=True, on_step=False, on_epoch=True)
#             self.log(f"{stage}_cat_prec", category_prec, prog_bar=True)
#             self.log(f"{stage}_cat_recall", category_recall, prog_bar=True)
#             self.log(f"{stage}_cat_f1", category_f1, prog_bar=True)
#             self.log(f"{stage}_ovr_prec", overall_prec, prog_bar=True)
#             self.log(f"{stage}_ovr_recall", overall_recall, prog_bar=True)
#             self.log(f"{stage}_ovr_f1", overall_f1, prog_bar=True)

#             # Log attention maps to wandb
#             if stage == "val" and self.global_step % 100 == 0 and attentions is not None:
#                 for layer, attn in enumerate(attentions):
#                     attn_map = attn[0].detach().cpu().numpy()
#                     for head in range(attn_map.shape[0]):
#                         attn_map_2d = attn_map[head].reshape(int(attn_map[head].size ** 0.5), -1)
#                         attn_map_normalized = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min() + 1e-8)
                        
#                         plt.figure(figsize=(10, 10))
#                         plt.imshow(attn_map_normalized, cmap='viridis')
#                         plt.colorbar()
#                         plt.title(f'Attention Map Layer {layer} Head {head}')
                        
#                         # Convert the matplotlib plot to a WandB image
#                         wandb_image = wandb.Image(plt)
#                         wandb.log({f"Attention Map Layer {layer} Head {head}": wandb_image})
#                         plt.close()

#     def training_step(self, batch, batch_idx):
#         if self.hparams.use_cutmix:
#             x, y = batch
#             y_hat = self(x)
#             loss = self.criterion(y_hat, y)
#         else:
#             x, y = batch
#             y_hat = self(x)
#             loss = self.base_criterion(y_hat, y.type(torch.float))
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.evaluate(batch, "val")

#     def test_step(self, batch, batch_idx):
#         self.evaluate(batch, "test")

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.learning_rate,
#             betas=(0.9, 0.999),
#             weight_decay=self.hparams.weight_decay,
#         )
#         lr_scheduler_dict = {
#             "scheduler": OneCycleLR(
#                 optimizer,
#                 self.hparams.learning_rate,
#                 epochs=self.trainer.max_epochs,
#                 steps_per_epoch=len(self.data.train_dataloader()),
#                 anneal_strategy="linear",
#             ),
#             "interval": "step",
#         }
#         return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}


# import os
# import numpy as np
# import scipy.ndimage
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torchmetrics.functional as tf
# from torch.optim.lr_scheduler import OneCycleLR
# from torchvision import transforms
# from query2label.models.query2label import Query2Label
# from query2label.models.asymmetric_loss import AsymmetricLoss
# from query2label.models.cutmix import CutMixCriterion
# import wandb

# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torchmetrics.functional as tf
# from torch.optim.lr_scheduler import OneCycleLR
# from q2l_labeller.models.query2label import Query2Label
# from q2l_labeller.losses.asymmetric_loss import AsymmetricLoss
# from q2l_labeller.losses.two_way_loss import TwoWayLoss
# from q2l_labeller.losses.cutmix_loss import CutMixCriterion
# import numpy as np
# import matplotlib.pyplot as plt
# import wandb  # Uncomment if using wandb

def calculate_map_at_k(y_true, y_pred, k=20):
    """Calculate mAP@k for multi-label classification."""
    ap_scores = []

    for true_labels, pred_scores in zip(y_true, y_pred):
        top_k_indices = pred_scores.argsort()[-k:][::-1]
        num_relevant = 0
        ap_sum = 0

        for i, index in enumerate(top_k_indices):
            if true_labels[index] == 1:
                num_relevant += 1
                ap_sum += num_relevant / (i + 1)

        if num_relevant > 0:
            ap_scores.append(ap_sum / min(num_relevant, k))
        else:
            ap_scores.append(0.0)

    return np.mean(ap_scores)

class Query2LabelTrainModule(pl.LightningModule):
    def __init__(
        self,
        data,
        backbone_desc,
        conv_out_dim,
        hidden_dim,
        num_encoders,
        num_decoders,
        num_heads,
        batch_size,
        image_dim,
        learning_rate,
        momentum,
        weight_decay,
        n_classes,
        thresh=0.5,
        use_cutmix=False,
        use_pos_encoding=False,
        loss="BCE",
        use_seathru=False,  # Add this parameter
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "data"])
        self.data = data
        self.auroc = tf.auroc
        self.model = Query2Label(
            model=backbone_desc,
            conv_out=conv_out_dim,
            num_classes=n_classes,
            hidden_dim=hidden_dim,
            nheads=num_heads,
            encoder_layers=num_encoders,
            decoder_layers=num_decoders,
            use_pos_encoding=use_pos_encoding,
        )
        if loss == "BCE":
            self.base_criterion = nn.BCEWithLogitsLoss()
        elif loss == "ASL":
            self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
        elif loss == "mll":
            self.base_criterion = TwoWayLoss()

        self.criterion = CutMixCriterion(self.base_criterion)

    # def forward(self, x, return_attention=False):
    #     return self.model(x, return_attention)
    def forward(self, x):
        x = self.model(x)
        return x

    def evaluate(self, batch, stage=None):
        # x, y = batch
        # y_hat, attentions = self(x, return_attention=True)
        # loss = self.base_criterion(y_hat, y.type(torch.float))

        # rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))
        x, y = batch
        y_hat = self(x)
        loss = self.base_criterion(y_hat, y.type(torch.float))

        rmap = tf.retrieval_average_precision(y_hat, y.type(torch.int))

        category_prec = tf.precision(
            y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh, multiclass=False)
        category_recall = tf.recall(
            y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh, multiclass=False)
        category_f1 = tf.f1_score(
            y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes,
            threshold=self.hparams.thresh, multiclass=False)

        overall_prec = tf.precision(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)
        overall_recall = tf.recall(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)
        overall_f1 = tf.f1_score(
            y_hat, y.type(torch.int), threshold=self.hparams.thresh, multiclass=False)

        roc_auc = tf.auroc(y_hat, y.type(torch.int), average="macro", num_classes=self.hparams.n_classes)
        mAP = tf.average_precision(y_hat, y.type(torch.int), average="macro")
        y_hat_np = y_hat.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        map_20 = calculate_map_at_k(y_np, y_hat_np, k=20)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_rmap", rmap, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_cat_prec", category_prec, prog_bar=True)
            self.log(f"{stage}_cat_recall", category_recall, prog_bar=True)
            self.log(f"{stage}_cat_f1", category_f1, prog_bar=True)
            self.log(f"{stage}_ovr_prec", overall_prec, prog_bar=True)
            self.log(f"{stage}_ovr_recall", overall_recall, prog_bar=True)
            self.log(f"{stage}_ovr_f1", overall_f1, prog_bar=True)
            self.log(f"{stage}_roc_auc", roc_auc, prog_bar=True)
            self.log(f"{stage}_mAP", mAP, prog_bar=True)
            self.log(f"{stage}_mAP@20", map_20, prog_bar=True)

            # # Log attention maps to wandb
            # if stage == "val" and self.global_step % 100 == 0 and attentions is not None:
            #     for layer, attn in enumerate(attentions):
            #         attn_map = attn[0].detach().cpu().numpy()
            #         for head in range(attn_map.shape[0]):
            #             attn_map_2d = attn_map[head].reshape(int(attn_map[head].size ** 0.5), -1)
            #             attn_map_normalized = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min() + 1e-8)
                        
            #             plt.figure(figsize=(10, 10))
            #             plt.imshow(attn_map_normalized, cmap='viridis')
            #             plt.colorbar()
            #             plt.title(f'Attention Map Layer {layer} Head {head}')
                        
            #             # Convert the matplotlib plot to a WandB image
            #             wandb_image = wandb.Image(plt)
            #             wandb.log({f"Attention Map Layer {layer} Head {head}": wandb_image})
            #             plt.close()

    def training_step(self, batch, batch_idx):
        if self.hparams.use_cutmix:
            x, y = batch
            y_hat = self(x)
            # y1, y2, lam = y
            loss = self.criterion(y_hat, y)

        else:
            x, y = batch
            y_hat = self(x)
            loss = self.base_criterion(y_hat, y.type(torch.float))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.data.train_dataloader()),
                anneal_strategy="cos",
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
