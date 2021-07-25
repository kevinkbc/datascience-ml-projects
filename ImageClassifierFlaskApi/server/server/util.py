from torchvision import transforms
import json

class Preprocessor:
    def __init__(self):
        self.preprocessing = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    def run(self, image):
        return self.preprocessing(image)

# Get the label according to the index best ranked
def getLabel(index):
    class_idx = json.load(open("./labels/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return idx2label[index]
