import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class TimeSeriesClassifier(pl.LightningModule):
    def __init__(self,
                 encoder,
                 clf,
                 learning_rate,
                 n_classes,
                 learning_rate_clf=None,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 input_shape=None,
                 l2_penalty=0.0,
                 momentum=0.9,
                 log_gradients=False,
                 optimizer_name='adam'):
        super(TimeSeriesClassifier, self).__init__()
        self.clf = clf
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        if learning_rate_clf:
            self.learning_rate_clf = learning_rate_clf
        else:
            self.learning_rate_clf = learning_rate
        self.loss_fn = loss_fn
        if input_shape is None:
            input_shape = (34, 600)
        self.example_input_array = torch.randn(16, *input_shape)
        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.log_gradients = log_gradients
        self.optimizer_name = optimizer_name.lower()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def forward(self, x):
        return self.clf(self.encoder(x))

    def custom_histogram_adder(self):
        if self.logger is not None:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(
                    name, params, self.current_epoch)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([
                {'params': self.encoder.parameters(),
                 'lr': self.learning_rate,
                 'l2_penalty': self.l2_penalty},
                {'params': self.clf.parameters(),
                 'lr': self.learning_rate_clf,
                 'l2_penalty': self.l2_penalty}
            ])
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': self.encoder.parameters(),
                 'lr': self.learning_rate,
                 'l2_penalty': self.l2_penalty,
                 'momentum': self.momentum},
                {'params': self.clf.parameters(),
                 'lr': self.learning_rate_clf,
                 'l2_penalty': self.l2_penalty,
                 'momentum': self.momentum}
            ])
        return {
            'optimizer': optimizer
        }

    def __step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        L = self.loss_fn(target=y, input=y_pred)
        acc = accuracy(preds=y_pred,
                       target=y,
                       task='multiclass',
                       top_k=1,
                       num_classes=self.n_classes)

        return {'loss': L, 'acc': acc}

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.training_step_outputs.append(output)

        return output

    def validation_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()

        self.history['loss'].append(avg_loss)
        self.history['acc'].append(avg_acc)

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "Loss/Train", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar(
                "Accuracy/Train", avg_acc, self.current_epoch)

        # Logs histograms
        if self.log_gradients:
            self.custom_histogram_adder()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "Loss/Validation", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar(
                "Accuracy/Validation", avg_acc, self.current_epoch)

        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def on_test_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()

        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        self.log_dict(results)
        return results
