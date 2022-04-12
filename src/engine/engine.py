import tqdm
import torch
import gc
from src.utils.utils import printDash
from src.data.dataset import PmatDataset, PseudoDataset
from src.engine.engine_tools import getOptimizer, getSchedu
from src.loss.am_softmax import AM_Softmax
from src.model.model import EfficientNet


class Engine:
    def __init__(self, cfg, list_model):
        self.teacher_model = None
        self.student_model = None
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_fn = AM_Softmax()
        self.list_model = list_model
        # self.optimizer = getOptimizer(self.cfg["optimizer"],
        #                             self.model.parameters(),
        #                             self.cfg["learning_rate"],
        #                             self.cfg["weight_decay"])
        # self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

    def train(self, train_data, val_data):

        for i in range(len(self.list_model)):
            if i == 0:
                self.teacher_model = EfficientNet.from_pretrain(self.list_model[i]).to(self.device)
                train_dataset = PmatDataset(train_data)
                val_dataset = PmatDataset(val_data)
                self.train_model(self.teacher_model, train_dataset, val_dataset, save_path=self.cfg["teacher_path"])

            self.student_model = EfficientNet.from_pretrain(self.list_model[i + 1]).to(self.device)
            pseudo_train_dataset = PseudoDataset(self.teacher_model, self.device,
                                                 train_data, self.cfg["teacher_path"], soft=True)
            pseudo_val_dataset = PseudoDataset(self.teacher_model, self.device,
                                               val_data, self.cfg["teacher_path"], soft=False)
            self.train_model(self.student_model, pseudo_train_dataset, pseudo_val_dataset,
                             save_path=self.cfg["student_path"])

    def train_model(self, model, train_dataset, val_dataset, save_path):
        for epoch in range(self.cfg["num_epoch"]):
            # self.onTrainStep(train_loader, epoch)
            # self.onValidation(val_loader, epoch)
            self.save_checkpoint(model, save_path)
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


