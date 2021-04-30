from SimpleBaselineNet import SimpleBaselineNet
from experiment_runner_base import ExperimentRunnerBase
from celeba_dataset import CelebADataset
import argparse
from torchvision import transforms
from PIL import Image
import torch
import torchvision



def test(img_dir, model_dir):
    # transform
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    # Load Net and pretrained model 
    model = SimpleBaselineNet()
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval() 

    # Load image
    image = transform(Image.open(img_dir).convert("RGB"))

    # Predict height
    predicted_height = model(image.unsqueeze(0)).detach().cpu().numpy()

    if predicted_height <= 1.6:
        print("Predicted height: ", predicted_height, "meters\n", "Damn you short as hell, drink some milk!")
    elif predicted_height > 1.6 and predicted_height <= 1.8:
        print("Predicted height: ", predicted_height, "meters\n", "Alright, not bad, not bad at all")
    elif predicted_height > 1.8:
        print("Predicted height: ", predicted_height, "meters\n", "Hows the weather up there?")

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HeightNet.')
    parser.add_argument('--img_dir', type=str, default="data/test.jpg")
    parser.add_argument('--model_dir', type=str, default="models/epoch29step63000.pth")
    args = parser.parse_args()
    test(args.img_dir, args.model_dir)

