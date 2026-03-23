###########################################################################
#CGNet: A Light-weight Context Guided Network for Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from climatenet.utils.utils import Config
from os import path
import pathlib

class CGNet():
    '''
    The high-level CGNet class. 
    This allows training and running CGNet without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : CGNetModule
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    '''

    def __init__(self, config: Config = None, model_path: str = None):
    
        if config is not None and model_path is not None:
            raise ValueError('''Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model.''')

        if config is not None:
            # Create new model
            self.config = config
            self.network = CGNetModule(
                classes=len(self.config.labels),
                channels=len(list(self.config.fields)),
                dropout_flag=self.config.dropout_flag).cuda()
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = CGNetModule(
                classes=len(self.config.labels),
                channels=len(list(self.config.fields)),
                dropout_flag=self.config.dropout_flag).cuda()
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)        
        
    def train(self, train_dataset: ClimateDatasetLabeled,
            val_dataset: ClimateDatasetLabeled):

        collate = ClimateDatasetLabeled.collate

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=collate,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=collate,
            shuffle=False,
            num_workers=0
        )

        # Storage for curves
        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        train_mean_ious = []
        val_mean_ious = []

        train_ious_per_class_history = []
        val_ious_per_class_history = []

        train_precisions_history = []
        val_precisions_history = []

        train_recalls_history = []
        val_recalls_history = []

        train_specificities_history = []
        val_specificities_history = []

        for epoch in range(1, self.config.epochs + 1):

            print(f"\n===== Epoch {epoch} =====")

            ################################
            # TRAINING
            ################################
            self.network.train()

            epoch_loss = 0.0
            num_batches = 0
            train_cm = np.zeros((3, 3))

            for features, labels in tqdm(train_loader):

                # Convert to tensor
                features = torch.tensor(features.values)
                labels = torch.tensor(labels.values)

                # Remove extra dimensions if needed
                if features.dim() == 5:
                    features = features.squeeze(1)
                if features.dim() == 4 and features.shape[1] == 1:
                    features = features.squeeze(1)
                if labels.dim() == 4:
                    labels = labels.squeeze(1)

                features = features.cuda()
                labels = labels.cuda()

                # Forward
                outputs = torch.softmax(self.network(features), dim=1)

                loss = jaccard_loss(outputs, labels)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Update confusion matrix
                predictions = torch.argmax(outputs, dim=1)
                train_cm += get_cm(predictions, labels, 3)

            # Compute epoch metrics
            epoch_loss /= num_batches
            train_losses.append(epoch_loss)

            train_accuracy = np.trace(train_cm) / train_cm.sum()
            train_accuracies.append(train_accuracy)

            train_ious = get_iou_perClass(train_cm)
            train_mean_iou = train_ious.mean()
            train_mean_ious.append(train_mean_iou)
            train_ious_per_class_history.append(train_ious.tolist())

            total = train_cm.sum()
            train_prec, train_rec, train_spec = [], [], []
            for k in range(3):
                TP = train_cm[k, k]
                FP = train_cm[:, k].sum() - TP
                FN = train_cm[k, :].sum() - TP
                TN = total - (TP + FP + FN)
                train_prec.append(float(TP / (TP + FP + 1e-8)))
                train_rec.append(float(TP / (TP + FN + 1e-8)))
                train_spec.append(float(TN / (TN + FP + 1e-8)))
            train_precisions_history.append(train_prec)
            train_recalls_history.append(train_rec)
            train_specificities_history.append(train_spec)

            print(f"Train Loss: {epoch_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Train Mean IoU: {train_mean_iou:.4f}")
            print(f"Train IoU per class: {train_ious}")
            print(f"Train Recall (BG/TC/AR): {train_rec}")
            print(f"Train Precision (BG/TC/AR): {train_prec}")

            ################################
            # VALIDATION
            ################################
            self.network.eval()

            val_loss = 0.0
            val_batches = 0
            val_cm = np.zeros((3, 3))

            with torch.no_grad():
                for features, labels in val_loader:

                    features = torch.tensor(features.values)
                    labels = torch.tensor(labels.values)

                    if features.dim() == 5:
                        features = features.squeeze(1)
                    if features.dim() == 4 and features.shape[1] == 1:
                        features = features.squeeze(1)
                    if labels.dim() == 4:
                        labels = labels.squeeze(1)

                    features = features.cuda()
                    labels = labels.cuda()

                    outputs = torch.softmax(self.network(features), dim=1)

                    loss = jaccard_loss(outputs, labels)

                    val_loss += loss.item()
                    val_batches += 1

                    predictions = torch.argmax(outputs, dim=1)
                    val_cm += get_cm(predictions, labels, 3)

            val_loss /= val_batches
            val_losses.append(val_loss)

            val_accuracy = np.trace(val_cm) / val_cm.sum()
            val_accuracies.append(val_accuracy)

            val_ious = get_iou_perClass(val_cm)
            val_mean_iou = val_ious.mean()
            val_mean_ious.append(val_mean_iou)
            val_ious_per_class_history.append(val_ious.tolist())

            total = val_cm.sum()
            val_prec, val_rec, val_spec = [], [], []
            for k in range(3):
                TP = val_cm[k, k]
                FP = val_cm[:, k].sum() - TP
                FN = val_cm[k, :].sum() - TP
                TN = total - (TP + FP + FN)
                val_prec.append(float(TP / (TP + FP + 1e-8)))
                val_rec.append(float(TP / (TP + FN + 1e-8)))
                val_spec.append(float(TN / (TN + FP + 1e-8)))
            val_precisions_history.append(val_prec)
            val_recalls_history.append(val_rec)
            val_specificities_history.append(val_spec)

            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val Mean IoU: {val_mean_iou:.4f}")
            print(f"Val IoU per class: {val_ious}")
            print(f"Val Recall (BG/TC/AR): {val_rec}")
            print(f"Val Precision (BG/TC/AR): {val_prec}")
            print(f"Val Specificity (BG/TC/AR): {val_spec}")

        ################################
        # Return history for plotting
        ################################
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_mean_ious": train_mean_ious,
            "val_mean_ious": val_mean_ious,
            "train_ious_per_class": train_ious_per_class_history,
            "val_ious_per_class": val_ious_per_class_history,
            "train_precisions": train_precisions_history,
            "val_precisions": val_precisions_history,
            "train_recalls": train_recalls_history,
            "val_recalls": val_recalls_history,
            "train_specificities": train_specificities_history,
            "val_specificities": val_specificities_history,
        }

    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        '''Make predictions for the given dataset and return them as xr.DataArray'''
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values)

            if features.dim() == 5:
                features = features.squeeze(1)

            features = features.cuda()
        
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            coords = batch.coords
            del coords['variable']
            
            dims = [dim for dim in batch.dims if dim != "variable"]
            
            predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

        return xr.concat(predictions, dim='time')

    def predict_with_uncertainty(self, dataset: ClimateDataset, n_passes: int = 30):
        '''
        MC Dropout inference: run n_passes forward passes with dropout active,
        return mean prediction and per-pixel uncertainty (variance).

        Returns
        -------
        mean_preds : xr.DataArray  — argmax of mean softmax, shape (time, lat, lon)
        uncertainty : xr.DataArray — mean variance across classes, shape (time, lat, lon)
        '''
        self.network.train()  # keep dropout active
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)

        all_mean_preds = []
        all_uncertainty = []

        for batch in tqdm(loader):
            features = torch.tensor(batch.values)
            if features.dim() == 5:
                features = features.squeeze(1)
            features = features.cuda()

            # Collect n_passes softmax outputs: list of (B, C, H, W)
            passes = []
            with torch.no_grad():
                for _ in range(n_passes):
                    out = torch.softmax(self.network(features), dim=1)
                    passes.append(out.cpu().numpy())

            # passes: (n_passes, B, C, H, W)
            passes = np.stack(passes, axis=0)
            mean_probs = passes.mean(axis=0)          # (B, C, H, W)
            variance = passes.var(axis=0).mean(axis=1) # (B, H, W) — avg var across classes

            preds = mean_probs.argmax(axis=1)          # (B, H, W)

            coords = batch.coords
            del coords['variable']
            dims = [d for d in batch.dims if d != 'variable']

            all_mean_preds.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))
            all_uncertainty.append(xr.DataArray(variance, coords=coords, dims=dims, attrs=batch.attrs))

        self.network.eval()
        return xr.concat(all_mean_preds, dim='time'), xr.concat(all_uncertainty, dim='time')

    def evaluate(self, dataset: ClimateDatasetLabeled):
        '''Evaluate on a dataset and return statistics'''
        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=0)

        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3,3))

        for features, labels in epoch_loader:
        
            features = torch.tensor(features.values)
            labels = torch.tensor(labels.values)

            if features.dim() == 5:
                features = features.squeeze(1)
            if features.dim() == 4 and features.shape[1] == 1:
                features = features.squeeze(1)

            if labels.dim() == 4:
                labels = labels.squeeze(1)

            features = features.cuda()
            labels = labels.cuda()
                
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            aggregate_cm += get_cm(predictions, labels, 3)

        print('Evaluation stats:')
        print(aggregate_cm)
        ious = get_iou_perClass(aggregate_cm)
        print('Evaluation IOUs: ', ious, ', Evaluation mean: ', ious.mean())

        total = aggregate_cm.sum()
        num_classes = aggregate_cm.shape[0]

        precisions = []
        recalls = []
        specificities = []
        f1_scores = []

        for k in range(num_classes):
            TP = aggregate_cm[k, k]
            FP = aggregate_cm[:, k].sum() - TP
            FN = aggregate_cm[k, :].sum() - TP
            TN = total - (TP + FP + FN)

            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1_scores.append(f1)

        print("Evaluate Precision per class:", np.array(precisions))
        print("Evaluate Recall per class:", np.array(recalls))
        print("Evaluate Specificity per class:", np.array(specificities))
        print("Evaluate F1 per class:", np.array(f1_scores))

        print("Evaluate Macro Precision:", np.mean(precisions))
        print("Evaluate Macro Recall:", np.mean(recalls))
        print("Evaluate Macro Specificity:", np.mean(specificities))
        print("Evaluate Macro F1:", np.mean(f1_scores))
        return {
        "confusion_matrix": aggregate_cm.tolist(),
        "ious_per_class (BKG, TC, AR)": ious.tolist(),
        "mean_iou": float(ious.mean()),
        "precision_per_class (BKG, TC, AR)": np.array(precisions).tolist(),
        "recall_per_class (BKG, TC, AR)": np.array(recalls).tolist(),
        "specificity_per_class (BKG, TC, AR)": np.array(specificities).tolist(),
        "f1_per_class (BKG, TC, AR)": np.array(f1_scores).tolist(),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(np.mean(f1_scores))
        }

    def save_model(self, save_path: str):
        '''
        Save model weights and config to a directory.
        '''
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

        # save weights and config
        self.config.save(path.join(save_path, 'config.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights.pth'))

    def load_model(self, model_path: str):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = CGNetModule(classes=len(self.config.labels), channels=len(list(self.config.fields))).cuda()
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))


class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """
    def __init__(self, classes=19, channels=4, M=3, N= 21, dropout_flag = False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(channels, 32, 3, 2)      # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)                          
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)      

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + channels)
        
        #stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 + channels, 64, dilation_rate=2,reduction=8)  
        self.level2 = nn.ModuleList()
        for i in range(0, M-1):
            self.level2.append(ContextGuidedBlock(64 , 64, dilation_rate=2, reduction=8))  #CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)
        
        #stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + channels, 128, dilation_rate=4, reduction=16) 
        self.level3 = nn.ModuleList()
        for i in range(0, N-1):
            self.level3.append(ContextGuidedBlock(128 , 128, dilation_rate=4, reduction=16)) # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d')!= -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        
        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out
      
   
