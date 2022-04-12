import tqdm
import torch
import gc
from src.utils.utils import printDash
from src.engine.engine_tools import getOptimizer, getSchedu
from src.loss.am_softmax import AM_Softmax
from model.model import EfficientNet

class Engine():
    def __init__(self, cfg, list_model):
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_fn = AM_Softmax()
        self.list_model = list_model 
        # self.optimizer = getOptimizer(self.cfg["optimizer"],
        #                             self.model,
        #                             self.cfg["learning_rate"],
        #                             self.cfg["weight_decay"])
        # self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)


    def train(self, train_loader, val_loader):
        for i in range(self.list_model):
            self.teacher_model = EfficientNet.from_pretrain(self.list_model[i]).to(self.device)
            self.student_model = EfficientNet.from_pretrain(self.list_model[i+1]).to(self.device)

        if i == 0:
            self.train_model(self.teacher_model, train_loader, val_loader)
        self.create
    
    def first_train_teacher(self, teacher_model, train_loader, val_loader):
        for epoch in range(self.cfg["num_epoch"]):
            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader, epoch)
        self.onTrainEnd()


    def onTrainEnd(self):
        del self.teacher_model
        del self.student_model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()