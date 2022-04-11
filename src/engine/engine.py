import tqdm
import torch
from src.engine.engine_tools import getOptimizer, getSchedu
from src.loss.am_softmax import AM_Softmax

class Engine():
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.loss_fn = AM_Softmax()
        self.optimizer = getOptimizer(self.cfg["optimizer"],
                                    self.model,
                                    self.cfg["learning_rate"],
                                    self.cfg["weight_decay"])
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)


    def train(self, train_loader, val_loader):
        pass