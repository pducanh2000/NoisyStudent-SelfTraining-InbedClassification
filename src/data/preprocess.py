from torchvision.transforms import ToPILImage, ToTensor, Normalize

def preprocess(image, posture):
    image = cv2.equalizeHist(image)
        
    image = ToPILImage()(image)
    image = image.convert('RGB')
    image = image.resize((112, 224))

    image = ToTensor()(image)
    image = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(image)

    return image, torch.tensor(posture, dtype=torch.long)