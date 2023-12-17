import dash
from dash import Dash, html, Input, Output, dcc, ALL
import STE_TCVAE as tc
from torchvision import transforms
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch

class ImageClass(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{idx + 1:06d}.png')
        image = read_image(img_path) / 255
        if self.transform:
            image = self.transform(image)
        return image


def load_basic_model(path, dataset_size = 182599, positive_w = torch.Tensor([202599/9818])):
    model = tc.STETCVAE(dataset_size, positive_w)
    model.load_state_dict(torch.load(path))
    return model.eval()


def make_image(array):
    return Image.fromarray((np.moveaxis(np.array(array), 0, -1) * 255).astype(np.uint8))


if __name__ == '__main__':
    model = load_basic_model('example/example_model.pt')

    tensor_trans = transforms.Compose([transforms.Resize((72, 60))])
    imfold = ImageClass('image_folder_path')

    # remove fully inactive latents from selection
    for param in model.parameters():
        if param.shape == torch.Size([7680, 200]):
            ml = list(np.where(abs(param.data).mean(0) > 0.08))[0]

    lock_buttons = [html.Button('lock', id={'type': 'lockbutton', 'index': f'button_{i}'}, n_clicks=0, className='button')
                    for i in range(len(ml))]


    def gen_sliders(values):
        sliders = []
        for i, dim in enumerate(ml):
            sliders.append(html.Div([
                html.Div([dim], className='dimname'),
                lock_buttons[i],
                dcc.Slider(-10, 10, value=values[i], id={'type': 'sliderlist', 'index': f'sliders_{dim}'},
                           className='slidersbox')],
                className='slider'
            ))
            return sliders


    slider_div = html.Div(children=gen_sliders([0]*len(ml)), id='slidersbox')

    app = Dash(__name__)
    app.layout = html.Div([
        html.Div(children=['Select an image to use as a basis:',
                           dcc.Input(id='image_input', value=0, type='number', min=0, max=202599, step=1)]),
        html.Div(id='images', children=[
            html.Div(children=html.Img(id='image_output', className='image'), clasName='imagecontainer'),
            html.Div(children=html.Img(id='image_output', className='static_decodec'), clasName='imagecontainer'),
            html.Div(children=html.Img(id='image_output', className='decoder_output'), clasName='imagecontainer'),
        ]),
        slider_div,

        html.Div(children=[], style={'display': 'none'}, id='latent_store'),
        html.Div(children=[0] * len(ml), style={'display': 'none'}, id='lock_codes')
    ])

    @app.callback(
        Output(component_id='image_output', component_property='src'),
        Output(component_id={'type': 'sliderlist', 'index': ALL}, component_property='value'),
        Output(component_id='latent_store', component_property='children'),
        Output(component_id='static_decoded', component_property='src'),
        Input(component_id='image_input', component_property='value'),
        Input(component_id='lock_codes', component_property='children'),
    )
    def show_image(value, codes):
        val = imfold[value]
        enc = model.encoder((torch.unsqueeze(val, dim=0)))[0][0]
        ini_z = enc.detach().cpu().numpy()[ml]
        z = []
        for i, code in enumerate(codes):
            if code:
                z.append(dash.no_update)
            else:
                z.append(ini_z[i])

        return (make_image(val), z, enc,
                make_image(np.squeeze(model.decoder(torch.unsqueeze(enc, dim=0)).detach().cpu().numpy())))

    @app.callback(
        Output(component_id='decoder_output', component_property='src'),
        Input(component_id={'type': 'sliderlist', 'index': ALL}, component_property='value'),
        Input(component_id='latent_store', component_property='children'),
    )
    def show_decoded_image(latent_sliders, latent_space):

        lspace = np.array(latent_space)
        lspace[ml] = latent_sliders
        lspace = torch.unsqueeze(torch.FloatTensor(lspace), dim=0)

        return make_image(np.squeeze(model.decoder(lspace).detach().cpu().numpy()))

    @app.callback(
        Output(component_id='lock_codes', component_property='children'),
        Output(component_id={'type': 'lockbutton', 'index': ALL}, component_property='style'),
        Output(component_id={'type': 'lockbutton', 'index': ALL}, component_property='children'),
        Input(component_id={'type': 'lockbutton', 'index': ALL}, component_property='n_clicks'),
    )
    def lock_sliders(buttons):
        return list(zip(*[(True, {'color': 'blue'}, 'unlock') if button % 2 else
                          (False, {'color': 'black'}, 'lock') for button in buttons]))


    app.run_server(debug=True)
