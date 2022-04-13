import torch
import gc
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from src.utils.utils import printDash
from src.data.dataset import PmatDataset, PseudoDataset
from src.engine.engine_tools import getOptimizer, getSchedu
from src.loss.am_softmax import AM_Softmax, CrossEntropyLabelSmooth
from src.model.model import EfficientNet


class Engine:
    def __init__(self, cfg, list_model):
        self.teacher_model = None
        self.student_model = None
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_fn = CrossEntropyLabelSmooth(num_classes=self.cfg["num_class"])
        self.list_model = list_model

    def train(self, train_data, val_data):

        for i in range(len(self.list_model)):
            if i == 0:
                # Train teacher model in first step
                self.teacher_model = EfficientNet.from_pretrain(self.list_model[i], num_class=self.cfg["num_class"])
                train_dataset = PmatDataset(train_data)
                val_dataset = PmatDataset(val_data)
                self.train_model(self.teacher_model, train_dataset, val_dataset, save_path=self.cfg["teacher_path"])

            # Create the pseudo labels for student training
            self.student_model = EfficientNet.from_pretrain(self.list_model[i + 1], num_class=self.cfg["num_class"])
            pseudo_train_dataset = PseudoDataset(self.teacher_model, self.device,
                                                 train_data, self.cfg["teacher_path"], soft=True)
            pseudo_val_dataset = PseudoDataset(self.teacher_model, self.device,
                                               val_data, self.cfg["teacher_path"], soft=False)

            # Train the student model
            self.train_model(self.student_model, pseudo_train_dataset, pseudo_val_dataset,
                             save_path=self.cfg["student_path"])

    def train_model(self, model, train_dataset, val_dataset, save_path):
        optimizer = getOptimizer(self.cfg["optimizer"], model.parameters(),
                                 self.cfg["learning_rate"], self.cfg["weight_decay"])
        scheduler = getSchedu(self.cfg["scheduler"], optimizer)
        train_loader = DataLoader(train_dataset, batch_size=self.cfg["batch_size"], num_workers=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg["batch_size"], num_workers=2, shuffle=True)
        model = model.to(self.device)
        max_test_acc = 0

        for epoch in range(self.cfg["num_epoch"]):
            print("********Epoch {}:********".format(epoch + 1))

            # Train step
            model.train()
            train_epoch_iterator = tqdm(train_loader,
                                        desc="Training (Step X) (loss=X.X)",
                                        bar_format="{l_bar}{r_bar}",
                                        dynamic_ncols=True, )
            train_losses = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            train_acc_scores = []

            for idx, (images, postures) in enumerate(train_epoch_iterator):
                images = images.to(self.device)
                postures = postures.to(self.device)
                predict = model(images)
                loss = self.loss_fn(postures, predict)
                train_epoch_iterator.set_description(
                    "Training (Step %d) (loss=%2.5f)" % (idx + 1, loss.item())
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, postures_pred = torch.max(predict, dim=1)
                _, postures_gt = torch.max(predict, dim=1)

                train_losses.append(loss.item())
                train_precisions.append(precision_score(postures_pred.cpu(), postures_gt.cpu(), average='macro', zero_division=0))
                train_recalls.append(recall_score(postures_pred.cpu(), postures_gt.cpu(), average='macro', zero_division=0))
                train_f1_scores.append(f1_score(postures_pred.cpu(), postures_gt.cpu(), average='macro', zero_division=0))
                train_acc_scores.append(accuracy_score(postures_pred.cpu(), postures_gt.cpu()))

            print('Train loss: ', sum(train_losses) / len(train_losses))
            print('Train score: Precision = %2.5f, Recall = %2.5f, F1_score = %2.5f, Acc_score = %2.5f' % (
                sum(train_precisions) / len(train_precisions),
                sum(train_recalls) / len(train_recalls), sum(train_f1_scores) / len(train_f1_scores),
                sum(train_acc_scores) / len(train_acc_scores)))
            scheduler.step()

            # Validation
            model.val()

        self.onTrainEnd(model)

    def onTrainEnd(self, model):
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()

    def save_checkpoint(self, model, save_path):
        torch.save(model.state_dict(), save_path)
