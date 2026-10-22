import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VinaFoodDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self._data = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self._data.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self._data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": label
        }

def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    data = {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long)
    }
    return data
