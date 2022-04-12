import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize


class PmatDataset(Dataset):
    def __init__(self, data, preprocess=None):
        super(PmatDataset, self).__init__()
        self.images = data["image"]
        self.postures = data["posture"].reshape(-1)

        self.preprocessing = preprocess

    def getitem(self, index):
        if self.preprocessing is not None:
            data_item = self.preprocessing(self.images[index], self.postures[index])
        else:
            data_item = self.transform(self.images[index], self.postures[index])
        return data_item

    def __len__(self):
        return len(self.postures)

    @staticmethod
    def transform(image, posture):

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image, torch.tensor(posture, dtype=torch.long)


class PseudoDataset(Dataset):
    def __init__(self, model, device, data, save_path, soft=True):
        super(PseudoDataset, self).__init__()
        model.load_state_dict(torch.load(save_path))
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.images = data["images"]
        self.postures = data["postures"].reshape(-1)
        self.soft = soft

    def getitem(self, index):
        image = ToTensor()(self.images[index])
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)
        input_tensor = image.view(1, image.shape[0], image.shape[1], image[2])
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
        if self.soft:
            output = torch.softmax(output, dim=1)
        else:
            _, output = torch.max(output, dim=1)
        return image, torch.tensor(output, dtype=torch.long)

    def __len__(self):
        return len(self.postures)
