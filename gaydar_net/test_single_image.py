from SimpleBaselineNet import SimpleBaselineNet
from facenet import FaceNet
from experiment_runner_base import ExperimentRunnerBase
from gay_dataset import GayDataset
import argparse
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt



def test(img_dir, vgg_model_dir, face_model_dir):
    # plot
    fig, ax = plt.subplots(1, 3)
    fig.suptitle('Saliency Maps w/ Different Networks')

    # transform
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load image
    og_img = Image.open(img_dir).convert("RGB")
    ax[0].set_title('Original Image')
    ax[0].imshow(og_img)
    ax[0].axis('off')


    for i, Model in enumerate([SimpleBaselineNet, FaceNet]):
        # Load Net and pretrained model
        model = Model()
        model_dir = vgg_model_dir if Model is SimpleBaselineNet else face_model_dir
        model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        model = model.to('cpu')
        model.eval()

        # Predict height & backprop
        image = transform(og_img).unsqueeze(0)
        image.requires_grad_() # need gradient wst input img for visualizing saliency map
        predicted_height = model(image)
        predicted_height.backward()
        predicted_height = predicted_height.detach().cpu().numpy()

        std = 10.034
        mean = 172.244
        predicted_height = 0.01 * (predicted_height * std + mean)

        if predicted_height <= 1.6:
            print("Predicted height: ", predicted_height, "meters\n", "Damn you short as hell, drink some milk!")
        elif predicted_height > 1.6 and predicted_height <= 1.8:
            print("Predicted height: ", predicted_height, "meters\n", "Alright, not bad, not bad at all")
        elif predicted_height > 1.8:
            print("Predicted height: ", predicted_height, "meters\n", "Hows the weather up there?")

        # saliency map - max across rgb channels
        saliency = torch.max(image.grad.data.abs(), dim=1)[0]
        ax[i+1].set_title("VGG Net" if Model is SimpleBaselineNet else "FaceNet")
        ax[i+1].imshow(saliency[0], cmap=plt.cm.hot)
        ax[i+1].axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HeightNet.')
    parser.add_argument('--img_dir', type=str, default="data/test.jpg")
    parser.add_argument('--vgg_model_dir', type=str, default="models/vgg.pth")
    parser.add_argument('--face_model_dir', type=str, default="models/facenet_finetune.pth")
    args = parser.parse_args()
    test(args.img_dir, args.vgg_model_dir, args.face_model_dir)
