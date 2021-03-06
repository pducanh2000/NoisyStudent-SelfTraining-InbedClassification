import torch.optim as optim


def getSchedu(schedu, optimizer):
    if schedu == "default":
        factor = 0.1
        patience = 5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor=factor, patience=patience, min_lr=0.000001)
    elif schedu == "steplr":
        step_size = 10
        gamma = 0.1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif schedu == "sgdr":
        T_0 = 5
        T_mult = 2
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=T_0,
                                                                   T_mult=T_mult)
    elif schedu == "multisteplr":
        milestones = [10, 30]
        gamma = 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=milestones,
                                                   gamma=gamma)
    return scheduler


def getOptimizer(optimize, parameters, learning_rate, weight_decay):
    if optimize == "Adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimize == "SGD":
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    return optimizer
