import torch
from torchvision.io import read_image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

def train(model, train_data, val_data, save_path, max_epochs = 200, early_stopping = 10, verbose = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters())
    best = (0, np.inf)

    val_size = len(val_data)

    for i, epoch in enumerate(range(max_epochs)):
        model.train()
        running_loss = 0
        running_ce = 0

        for j, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            optim.zero_grad()
            loss[0].backward()
            optim.step()

            running_loss += float(loss[0])
            running_ce += float(loss[7])

            if verbose:
                print(f"""batch {j} | epoch loss: {running_loss/(j+1):.1f} | batch_loss: {float(loss[0]):.1f}
                reconstruction loss: {float(loss[1]):.1f}| KL-loss: {float(loss[2]):.1f} | TC-loss: {float(loss[3]):.1f}
                beta: {float(loss[4]):.1f} | beta error: {float(loss[5]):.1f} | beta change: {float(loss[6]):.1f}
                epoch classification loss: {running_ce/(j+1):.1f} | batch classification loss: {float(loss[7]):.1f} 
                gamma: {float(loss[8]):.1f} | gamma error: {float(loss[9]):.1f} | gamma change: {float(loss[10]):.1f}              
                """)
            else:
                print(f"batch {j} | epoch loss: {running_loss/(j+1):.1f} \
                      | epoch classification loss: {running_ce/(j+1):.1f})", end='\r')

            print('evaluation')
            model.eval()
            full_loss = 0
            for x, y in val_data:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    _, loss = model(x, y)
                    full_loss += float(loss[0])
            print(f'\n epoch: {i} | validation loss: {full_loss/val_size:.1f}')
            if full_loss < best[1]:
                print('new best, saving model')
                best = (i, full_loss)
                torch.save(model.state_dict(), save_path)
            elif i-early_stopping > best[0]:
                print('early stopping')
                return None
    return model


class Image_dataset(Dataset):
    def __init__(self, label_path, img_dir, transform=None, target='Eyeglasses'):
        self.target = target
        self.labels = pd.read_csv(label_path, header=1, delimiter='\s+')

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{idx+1:06d}.png')
        image = read_image(img_path)/255

        label = self.labels.iloc[idx][self.target]

        if self.transform:
            image = self.transform(image)
        return image, label