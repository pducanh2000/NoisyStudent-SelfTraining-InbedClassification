from src.engine import Engine
from src.utils.utils import load_config
from src.data.utils import data_split

config = load_config(config_path="./config/config.yaml")
eng = Engine(cfg=config)

val_accs = []

for index_out in range(1, 14):
    print(f"################ Subject {index_out} #################")
    train_data, val_data = data_split(subject_out='S' + str(index_out))
    val_acc = eng.self_training(train_data, val_data)
    val_accs.append(val_acc)

print("Validation accuracy on each sbj: ", val_accs)
