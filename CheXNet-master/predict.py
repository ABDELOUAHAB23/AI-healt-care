import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import DenseNet121, CLASS_NAMES
import argparse
import numpy as np
import random

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def predict_image(model, image_path, device):
    """Make prediction for a single image"""
    # Image preprocessing
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert to probabilities
    probabilities = output.cpu().numpy()[0]
    
    # Sort predictions by probability
    sorted_indices = probabilities.argsort()[::-1]
    
    return [(CLASS_NAMES[idx], float(probabilities[idx])) for idx in sorted_indices]

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    parser = argparse.ArgumentParser(description='Predict chest conditions from X-ray images')
    parser.add_argument('image_path', type=str, help='Path to the X-ray image')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = DenseNet121(len(CLASS_NAMES)).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth.tar')
    
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']
        
        # Handle both DataParallel and single GPU cases
        if not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image_path}")
    predictions = predict_image(model, args.image_path, device)
    
    # Print results
    print("\nChest X-Ray Analysis Results:")
    print("=" * 60)
    print(f"{'Disease/Condition':<30} {'Probability':<10} {'Percentage':<10}")
    print("-" * 60)
    
    for condition, probability in predictions:
        percentage = probability * 100
        prob_str = f"{probability:.3f}"
        perc_str = f"{percentage:.1f}%"
        print(f"{condition:<30} {prob_str:<10} {perc_str:<10}")
    
    print("=" * 60)
    print("\nNote:")
    print("- Probabilities range from 0 (not detected) to 1 (highly likely)")
    print("- Multiple conditions may be present simultaneously")
    print("- Consult a medical professional for proper diagnosis")

if __name__ == '__main__':
    main()
