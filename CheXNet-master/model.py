# encoding: utf-8

"""
The main CheXNet model implementation (CPU version).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import multiprocessing
from torchvision.models import DenseNet121_Weights
import random

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Configuration
CKPT_PATH = os.path.join(os.path.dirname(__file__), 'model.pth.tar')
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = os.path.join(os.path.dirname(__file__), 'ChestX-ray14', 'images')
TEST_IMAGE_LIST = os.path.join(os.path.dirname(__file__), 'ChestX-ray14', 'labels', 'test_list.txt')
BATCH_SIZE = 64
NUM_WORKERS = min(8, multiprocessing.cpu_count())


def main():
    try:
        # Set seed for reproducibility
        set_seed(42)
        cudnn.benchmark = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # initialize and load the model
        model = DenseNet121(N_CLASSES).to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        
        model = model.to(device)

        if os.path.isfile(CKPT_PATH):
            print("=> loading checkpoint from", CKPT_PATH)
            try:
                checkpoint = torch.load(CKPT_PATH, map_location=device)
                # Handle both DataParallel and single GPU cases
                state_dict = checkpoint['state_dict']
                if not list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                print("=> loaded checkpoint successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Training from scratch...")
                
        else:
            print(f"=> no checkpoint found at {CKPT_PATH}")
            print("Training from scratch...")
        
        # Verify data directory exists
        if not os.path.exists(DATA_DIR):
            print(f"Error: Data directory not found at {DATA_DIR}")
            return

        if not os.path.exists(TEST_IMAGE_LIST):
            print(f"Error: Test image list not found at {TEST_IMAGE_LIST}")
            return

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])

        test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                      image_list_file=TEST_IMAGE_LIST,
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.TenCrop(224),
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                          transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                      ]))

        if len(test_dataset) == 0:
            print("Error: No images found in the dataset")
            return

        test_loader = DataLoader(dataset=test_dataset, 
                               batch_size=BATCH_SIZE,
                               shuffle=False, 
                               num_workers=NUM_WORKERS, 
                               pin_memory=True)

        # initialize the ground truth and output tensor
        gt = torch.FloatTensor().to(device)
        pred = torch.FloatTensor().to(device)

        # switch to evaluate mode
        model.eval()

        print(f"Processing {len(test_dataset)} images...")
        for i, (inp, target) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}")
                
            target = target.to(device)
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = inp.view(-1, c, h, w).to(device)
            with torch.no_grad():
                output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean), 0)

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {:.3f}'.format(AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))

    except Exception as e:
        print(f"An error occurred: {e}")


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Modified DenseNet121 with sigmoid output."""
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # Use weights parameter instead of pretrained
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)


if __name__ == '__main__':
    main()
