import torch
import torch.nn as nn
import argparse
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits

from skimage.transform import resize
from torchvision import transforms as pth_transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

from numpy import save
import os



parser = argparse.ArgumentParser("Visualize Self-Attention maps")
parser.add_argument(
    "--patch_size", default=8, type=int, help="Patch resolution of the model."
)
parser.add_argument(
    "--dataset", default='DUTS', type=str, help="Name of the dataset."
)
parser.add_argument(
    "--mode", default='train', type=str, help="train or test."
)

args = parser.parse_args()
#####################################################dataset#######################################################
# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ImageDataset:
    def __init__(self, image_path, resize=None):

        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        if resize is not None:
            transform_resize = pth_transforms.Compose(
                [

                    pth_transforms.Resize(resize),
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

                ]
            )
            img = transform_resize(img)
            self.img_size = list(img.shape[-1:-3:-1])
        else:
            img = transform(img)
            self.img_size = list(img.shape[-1:-3:-1])
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return Image.open(self.image_path).convert("RGB").resize(self.img_size)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#[im_id, inp] = dataset.dataloader

###################################################model insert############################################
# model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
# state_dict = torch.load('/home/xubinwei/code/wsod2/image_processing/TOKEN/TokenCut/dino_deitsmall16_pretrain.pth')


model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
state_dict = torch.load('dino_deitsmall8_pretrain.pth')

#state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/" + url)
msg = model.load_state_dict(state_dict, strict=True)

model.eval()
model.cuda()


###########################数据加载########################################################



LIST = '../VRF/data/' + args.dataset + '/image'
#LIST = '/home/xubinwei/code/wsod/v1/data/DUTS-TE/image'
#LIST = '/home/xubinwei/code/wsod/v1/data/COD/TrainDataset/Imgs'
#LIST = '/home/xubinwei/code/wsod/v1/data/COD/TestDataset/CHAMELEON/Imgs'
LIST_FILE = os.listdir(LIST)
print(LIST_FILE)


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()  # 记录开始时间
for img_name in LIST_FILE:
    #name = 'ILSVRC2012_test_00009105'
    name = img_name.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
    print(name)

    #name = 'ILSVRC2012_test_00000018'
    #print(dataset.dataloader[0][1].shape)
    dataset = ImageDataset(LIST + '/' + name + '.jpg', resize=(320, 320))
    img = dataset.dataloader[0][0]


    init_image_size = img.shape
    size_im = (
        img.shape[0],
        int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
        int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
    )

    paded = torch.zeros(size_im)
    paded[:, : img.shape[1], : img.shape[2]] = img
    img = paded
    img = img.cuda()

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    ##########################特征产生######################################################
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    attentions = model.get_last_selfattention(img[None, :, :, :])
    #print('attentions.shape', attentions.shape)

    scales = [args.patch_size, args.patch_size]

    # Dimensions
    nb_im = attentions.shape[0]  # Batch size
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens
    #print('nb_tokens', nb_tokens)

    qkv = (
        feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
    )
    #print('qkv.shape', qkv.shape)

    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

    feats = k
    feats = feats[0, 1:, :]


    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    #feats = feats.reshape(w_featmap, h_featmap)
    #$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$
    feats = F.normalize(feats, p=2) # 575 * 384
    #print(feats)


    ##################################################
    feats_40_40_384 = feats.reshape(40, 40, 384)
    feats_40_40_384 = np.array(feats_40_40_384.detach().cpu())


    if not os.path.exists(args.dataset + '_' + args.mode + '_token_similarity_40_40_384'):
        os.makedirs(args.dataset + '_' + args.mode + '_token_similarity_40_40_384')
    save(args.dataset + '_' + args.mode + '_token_similarity_40_40_384/' + name + '.npy', feats_40_40_384)

    # feats_20_20_384 = feats.reshape(20, 20, 384)
    # feats_20_20_384 = np.array(feats_20_20_384.detach().cpu())
    # save('/home/xubinwei/code/wsod2/image_processing/TOKEN/TokenCut/duts_train_token_similarity_20_20_384/' + name + '.npy', feats_20_20_384)
    ##################################################


    # feats = feats @ feats.transpose(0, 1)
    #
    # feats = np.array(feats.detach().cpu())

    #save('/home/xubinwei/code/wsod2/image_processing/TOKEN/TokenCut/duts_train_token_similarity_flip_320/' + name + '.npy', feats)


end_event.record()
elapsed_time_ms = start_event.elapsed_time(end_event)
elapsed_time_sec = elapsed_time_ms / 1000

print("训练时间（毫秒）：", elapsed_time_ms)
print("训练时间（秒）：", elapsed_time_sec)



    #print('11')


