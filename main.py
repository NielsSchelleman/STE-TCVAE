import pandas as pd
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader


from basinhopper_utils import AbsoluteWRAcc
from basinhopping import BasinHopper
from torchvision import transforms
from train_model import Image_dataset, train
from STE_TCVAE import STETCVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(latent_space, log_path, evaluator, colnames=None, model=None):

    if colnames is not None:
        scores = evaluator.predict(latent_space, colnames, nlargest=20)
    else:
        assert model is not None, "must provide either a list of latent features or the model"
        params = model['classifier.linear1.module.weight']
        cols = np.argpartition(abs(params.cpu().detach().numpy()).mean(0), -5)[-5:]

        scores = evaluator.predict(latent_space, cols, nlargest=20)

    df = pd.DataFrame(scores).transpose()
    df.columns = ['splitpoints','score']
    df.to_csv(log_path, sep='|')
    return scores

if __name__ == "__main__":

    transform = transforms.Compose([transforms.Resize((72, 60))])
    target = 'Eyeglasses'
    image_folder_path = 'example/images/'
    target_path = 'example/attributes_CelebA.txt'
    dataset = Image_dataset(label_path=target_path, img_dir=image_folder_path, transform=transform, target=target)
    true_pos = sum(dataset.labels[target] > 0)

    data_load = DataLoader(dataset, batch_size=32)
    train_data, val_data = torch.utils.data.random_split(data_load.dataset, [182599, 20000])
    train_data = DataLoader(train_data, batch_size=32, shuffle=True)
    val_data = DataLoader(val_data, batch_size=32)

    beta_vae = STETCVAE(dataset_size=182599, positive_weight=torch.Tensor([(202599-true_pos)/true_pos]))
    model = train(model=STETCVAE, train_data=train_data, val_data=val_data, save_path='example/example_model.pt')

    model = beta_vae.load_state_dict(torch.load('example/example_model.pt'))
    model.eval()
    f = open('example/latent_space.csv', 'w')
    writer = csv.writer(f)
    for i, (x, y) in enumerate(data_load):
        latents, _ = model.encoder(x.to(device))
        for latent in latents.detach().cpu().numpy():
            writer.writerow(latent)
    f.close()

    latent_space = pd.read_csv('example/latent_space.csv', header=None)

    # initialze quality measure
    target_size = 202599
    total_acc =true_pos / target_size
    phi = AbsoluteWRAcc(target_size, total_acc)

    optimizer = BasinHopper(target, phi)
    evaluate(latent_space=latent_space, log_path='example/beams.csv', evaluator=optimizer, model=model)

