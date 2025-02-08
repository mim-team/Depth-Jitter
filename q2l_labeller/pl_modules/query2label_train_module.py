import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAccuracy
import numpy as np
import warnings
from sklearn.metrics import average_precision_score

from q2l_labeller.data.cutmix import CutMixCriterion
from q2l_labeller.loss_modules.simple_asymmetric_loss import AsymmetricLoss
from q2l_labeller.loss_modules.TwoWayMultiLabelLoss import TwoWayLoss
from q2l_labeller.models.query2label import Query2Label

warnings.filterwarnings("ignore")

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
        dropout_rate=0.3
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "data"])

        self.data = data
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

        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Loss functions
        if loss == "BCE":
            self.base_criterion = nn.BCEWithLogitsLoss()
        elif loss == "ASL":
            self.base_criterion = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
        elif loss == "mll":
            self.base_criterion = TwoWayLoss()

        self.criterion = CutMixCriterion(self.base_criterion)

        # Metrics
        self.precision_metric = MultilabelPrecision(num_labels=n_classes, threshold=thresh, average="macro")
        self.recall_metric = MultilabelRecall(num_labels=n_classes, threshold=thresh, average="macro")
        self.f1_metric = MultilabelF1Score(num_labels=n_classes, threshold=thresh, average="macro")
        self.accuracy_metric = MultilabelAccuracy(num_labels=n_classes, threshold=thresh)

    def forward(self, x):
        x = self.model(x)
        return self.dropout(x)  # Apply dropout to the model output

    def calculate_map_at_k(self, y_true, y_pred, k=20):
        """Calculate mean Average Precision at K (mAP@k) for multi-label classification."""
        aps = []
        for true_labels, pred_scores in zip(y_true, y_pred):
            # Get top-k indices sorted by prediction confidence
            top_k_indices = pred_scores.argsort()[-k:][::-1]
            num_relevant = 0
            ap_sum = 0

            for i, index in enumerate(top_k_indices):
                if true_labels[index] == 1:
                    num_relevant += 1
                    ap_sum += num_relevant / (i + 1)

            if num_relevant > 0:
                aps.append(ap_sum / min(num_relevant, k))
            else:
                aps.append(0.0)

        return np.mean(aps)

    def calculate_map(self, y_true, y_pred):
        """Calculate mean Average Precision (mAP) and class-wise AP using scikit-learn."""
        aps = []
        class_aps = {}
        for label in range(y_true.shape[1]):
            if np.sum(y_true[:, label]) > 0:  # Ensure there are positives for the class
                ap = average_precision_score(y_true[:, label], y_pred[:, label])
                aps.append(ap)
                class_aps[f"class_{label}"] = ap
            else:
                class_aps[f"class_{label}"] = 0.0
        return np.mean(aps), class_aps

    def evaluate(self, batch, stage=None):
        self.model.eval()
        x, y = batch
        y_hat = self(x)

        # Compute loss
        loss = self.base_criterion(y_hat, y.type(torch.float))

        # Metrics
        # precision = self.precision_metric(y_hat, y.type(torch.int))
        # recall = self.recall_metric(y_hat, y.type(torch.int))
        # f1_score = self.f1_metric(y_hat, y.type(torch.int))
        # accuracy = self.accuracy_metric(y_hat, y.type(torch.int))

        # Compute mAP and mAP@20
        y_hat_np = y_hat.sigmoid().detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        mAP, class_aps = self.calculate_map(y_np, y_hat_np)
        map_at_20 = self.calculate_map_at_k(y_np, y_hat_np, k=20)

        if stage:
            # Log overall metrics
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_mAP", mAP, prog_bar=True)
            self.log(f"{stage}_mAP@20", map_at_20, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        y_hat = self(x)
        loss = self.base_criterion(y_hat, y.type(torch.float))
        self.log("train_loss", loss, prog_bar=True)
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
                div_factor=25,  # Start with a very small LR
                final_div_factor=1e4,  # Reduce LR to a very small value
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
