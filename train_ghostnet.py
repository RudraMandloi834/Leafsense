import os
import timm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from tqdm import tqdm

# ---------------- GPU ----------------

torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# ---------------- LOAD CSV ----------------

master = pd.read_csv(
    "dataset/final_master_labels.csv"
)

master = master.fillna(0)

print(master.shape)

# ---------------- LABELS ----------------

label_cols = [

    c for c in master.columns[1:]

    if c != "num_labels"
]

print(label_cols)

print("Total Classes:", len(label_cols))

# ---------------- SPLIT ----------------

train_df, temp_df = train_test_split(
    master,
    test_size=0.3,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.3333,
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ---------------- TRANSFORMS ----------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),

    transforms.GaussianBlur(3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- DATASET ----------------

class PlantDataset(Dataset):

    def __init__(self, df, image_dir, transform=None):

        self.df = df

        self.image_dir = image_dir

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[label_cols].values.astype("float32")
        )

        return image, labels

# ---------------- DATASETS ----------------

IMAGE_DIR = "dataset/all_images"

train_dataset = PlantDataset(
    train_df,
    IMAGE_DIR,
    transform
)

val_dataset = PlantDataset(
    val_df,
    IMAGE_DIR,
    transform
)

test_dataset = PlantDataset(
    test_df,
    IMAGE_DIR,
    transform
)

# ---------------- DATALOADERS ----------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ---------------- MODEL ----------------

model = timm.create_model(
    "ghostnetv2_100",
    pretrained=True,
    num_classes=len(label_cols)
)

model = model.to(device)

print(model)

# ---------------- LOSS + OPTIMIZER ----------------

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# ---------------- TRAINING ----------------

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    train_correct = 0

    train_total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = (
            torch.sigmoid(outputs) > 0.5
        ).float()

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.numel()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

    train_acc = (
        train_correct / train_total
    )

    # ---------------- VALIDATION ----------------

    model.eval()

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            val_correct += (
                preds == labels
            ).sum().item()

            val_total += labels.numel()

    val_acc = (
        val_correct / val_total
    )

    print(
        f"\nEpoch {epoch+1}"
        f"\nLoss: {total_loss/len(train_loader):.4f}"
        f"\nTrain Acc: {train_acc:.4f}"
        f"\nVal Acc: {val_acc:.4f}"
    )

# ---------------- TEST ----------------

model.eval()

test_correct = 0

test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        preds = (
            torch.sigmoid(outputs) > 0.5
        ).float()

        test_correct += (
            preds == labels
        ).sum().item()

        test_total += labels.numel()

test_acc = (
    test_correct / test_total
)

print("\nFinal Test Accuracy:", test_acc)

# ---------------- SAVE MODEL ----------------

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/ghostnet_realistic.pth"
)

print("GhostNet Saved Successfully")