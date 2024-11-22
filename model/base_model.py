from typing import Optional, Callable, Dict, Any
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim as Optimizer
import logging
import time

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel:
    """
    Base class for transfer learning models
    
    Parameters
    ----------
    n_classes : int
        the new number of classes
    device: Optional[str] 'cuda' or 'cpu', default(None)
            if None: cuda will be used if it is available
    """
    
    def __init__(self, n_classes: int, device: Optional[str] = None):
        self.n_classes = n_classes
        self.model = None
        
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _freeze_all_layers(self) -> None:
        """
        freeze all layers in the model
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def _train_one_epoch(self, train_loader: DataLoader,
                        optimizer: Optimizer,
                        criterion: Callable,
                        valid_loader: DataLoader = None,
                        epoch: int = 0,
                        each_batch_idx: int = 300) -> None:
        """
        Train model for one epoch
        """
        train_loss = 0
        data_size = 0

        for batch_idx, sample_batched in enumerate(train_loader):
            data, label = sample_batched['image'], sample_batched['label']
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device)

            optimizer.zero_grad()
            pred_prob = self.model(data)
            loss = criterion(pred_prob, label)
            loss.backward()

            train_loss += loss.item()
            data_size += label.size(0)

            optimizer.step()

            if batch_idx % each_batch_idx == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.sampler.indices),
                    100. * (batch_idx * len(data)) / len(train_loader.sampler.indices),
                    loss.item()))
                    
        if valid_loader:
            acc = self.evaluate(test_loader=valid_loader)
            print('Accuracy on the valid dataset {}'.format(acc))

        print('====> Epoch: {} Average loss: {:.4f}'.
              format(epoch, train_loss / data_size))

    def train(self, epochs: int, train_loader: DataLoader,
              valid_loader: DataLoader = None) -> None:
        """
        Train model for several epochs
        """
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self._train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                valid_loader=valid_loader,
                epoch=epoch
            )

    def evaluate(self, test_loader: DataLoader) -> float:
        """
        Calculate model accuracy on test data
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, labels = sample_batched['image'], sample_batched['label']
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """
        Run the inference pipeline on the test_loader data
        """
        self.model.eval()
        self.model.to(self.device)
        predict_results = np.empty(shape=(0, self.n_classes))
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, _ = sample_batched['image'], sample_batched['label']
                data = data.to(self.device).float()
                outputs = self.model(data)
                outputs = softmax(outputs)
                predict_results = np.concatenate(
                    (predict_results, outputs.cpu().numpy()))
        return predict_results

    def evaluate_per_class(self, test_loader: DataLoader) -> (float,float,dict):
        self.model.eval()
        all_labels = []
        all_predictions = []
        confusion_mat = torch.zeros(self.n_classes,self.n_classes)
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                if isinstance(sample_batched, dict):
                    data, labels = sample_batched['image'], sample_batched['label']
                else:
                    data, labels = sample_batched
                
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_mat[t.long(), p.long()] += 1
                #all_labels.extend(labels.cpu().numpy())
                #all_predictions.extend(predicted.cpu().numpy())

        confusion_mat = confusion_mat.cpu().numpy()
        #all_labels = np.array(all_labels)
        #all_predictions = np.array(all_predictions)

        # 1. Calculate confusion matrix
        #conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(self.n_classes))

        # 2. Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_mat)

        # 3. Calculate metrics
        results = {}
        start_time = time.time()
        results['overall_accuracy'] = np.trace(confusion_mat) / np.sum(confusion_mat)
        results['per_class_precision'] = {}
        results['per_class_recall'] = {}
        results['per_class_false_alarm'] = {}
        
        
        precisions = []
        recalls = []
        false_alarms = []

        for i in range(self.n_classes):
            tp = confusion_mat[i, i]
            fp = np.sum(confusion_mat[:, i]) - tp
            fn = np.sum(confusion_mat[i, :]) - tp
            tn = np.sum(confusion_mat) - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            results['per_class_precision'][i] = precision
            precisions.append(precision)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results['per_class_recall'][i] = recall
            recalls.append(recall)
            false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            results['per_class_false_alarm'][i] = false_alarm 
            false_alarms.append(false_alarm)

        # Calculate standard deviations
        results['per_precision_sd'] = np.std(precisions)
        results['per_recall_sd'] = np.std(recalls)
        results['per_false_alarm_sd'] = np.std(false_alarms)

        # 4. Prepare results in JSON-like format
        
        results = {
            "overall_acc": float(results['overall_accuracy']),
            "per_class_precision": {k: float(v) for k, v in results['per_class_precision'].items()},
            "per_class_recall": {k: float(v) for k, v in results['per_class_recall'].items()},
            "per_class_falsealarm": {k: float(v) for k, v in results['per_class_false_alarm'].items()},
            "per_precision_sd": float(results['per_precision_sd']),
            "per_recall_sd": float(results['per_recall_sd']),
            "per_false_alarm_sd": float(results['per_false_alarm_sd'])
        }
        end_time = time.time()
        # Print results
        results['spend_time'] = end_time - start_time
        return results['overall_accuracy'], results['per_precision_sd'],results['per_class_precision']

"""
    def evaluate_per_class(self, test_loader: DataLoader) -> (float, float, dict):
        
        #Evaluate the model's accuracy for each class
       
        self.model.eval()
        correct_per_class = np.zeros(self.n_classes)
        total_per_class = np.zeros(self.n_classes)
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, labels = sample_batched['image'], sample_batched['label']
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    total_per_class[label.item()] += 1
                    if label.item() == prediction.item():
                        correct_per_class[label.item()] += 1

        per_class_accuracy = {}
        accuracies = []
        for i in range(self.n_classes):
            if total_per_class[i] > 0:
                acc = correct_per_class[i] / total_per_class[i]
                per_class_accuracy[i] = acc
                accuracies.append(acc)
            else:
                per_class_accuracy[i] = None

        accuracies = np.array(accuracies)
        sd = np.std(accuracies)
        overall_accuracy = total_correct / total_samples
        # [[1, 2], 
        #  [3, 4]]

        return overall_accuracy, sd, per_class_accuracy
        # print(overall_acc, per_class_precision, per_class_recall, per_calss_falsealarm) # json
        # {'overall_acc': 0.4, 'dfs'}
        # count[i, j]: ground truth:j, predicted: i, number
"""