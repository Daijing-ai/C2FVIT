import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from C2FViT_model import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm

import numpy as np




def dice(im1, atlas):
    unique_class = np.unique(atlas)           
    dice = 0
    num_count = 0
    for i in unique_class:
     
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice / num_count

model_path = r"D:\project\regis\Myproject\Model\C2FViT_affine_COM_pairwise_NCC_10000.pth"
savepath = r"D:\project\regis\Myproject\Test_Result"
fixed_path = r"D:\project\regis\Myproject\Data\Test_nii\image\img0438.nii.gz"
moving_path = r"D:\project\regis\Myproject\Data\Test_nii\image\img0439.nii.gz"
label_Fixed_path = r"D:\project\regis\Myproject\Data\Test_nii\label\seg0438.nii.gz"
label_Moving_path = r"D:\project\regis\Myproject\Data\Test_nii\label\seg0439.nii.gz"

com_initial = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)
# when num_stages=4
# model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15, 31], stride=[2, 4, 8, 16], num_classes=12,
#                       embed_dims=[256, 256, 256, 256],
#                       num_heads=[2, 2, 2, 2], mlp_ratios=[2, 2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                       attn_drop_rate=0., norm_layer=nn.Identity,
#                       depths=[4, 4, 4, 4], sr_ratios=[1, 1, 1, 1], num_stages=4, linear=False).cuda()

print(f"Loading model weight {model_path} ...")
model.load_state_dict(torch.load(model_path))
model.eval()

affine_transform = AffineCOMTransform().cuda()
init_center = Center_of_mass_initial_pairwise()

fixed_base = os.path.basename(fixed_path)
moving_base = os.path.basename(moving_path)

fixed_img_nii = nib.load(fixed_path)
header, affine = fixed_img_nii.header, fixed_img_nii.affine
fixed_img = fixed_img_nii.get_fdata()
fixed_img = np.reshape(fixed_img, (1,) + fixed_img.shape)    #(1, 160, 192, 224)

fixed_label_nii = nib.load(label_Fixed_path)
label_header, label_affine = fixed_label_nii.header, fixed_label_nii.affine
fixed_label = fixed_label_nii.get_fdata()
fixed_label = np.reshape(fixed_label, (1,) + fixed_label.shape)    #(1, 160, 192, 224)

moving_img = load_4D(moving_path)    #(1, 160, 192, 224)


img_B_label = load_4D(label_Moving_path)           #(1, 160, 192, 224)

fixed_img = min_max_norm(fixed_img)              
moving_img = min_max_norm(moving_img)
fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)                   #(1, 1, 160, 192, 224)
moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)                  #(1, 1, 160, 192, 224)
fixed_label = torch.from_numpy(fixed_label).float().to(device).unsqueeze(dim=0)               #(1, 1, 160, 192, 224)
moving_label =torch.from_numpy(img_B_label).float().to(device).unsqueeze(dim=0)               #(1, 1, 160, 192, 224)

with torch.no_grad():
        if com_initial:
            moving_img, init_flow = init_center(moving_img, fixed_img)
            X_label = F.grid_sample(moving_label, init_flow, mode="nearest", align_corners=True)           #(1, 1, 160, 192, 224)

        X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)        #(1, 1, 80, 96, 112)
        Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)        #(1, 1, 80, 96, 112)

        warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
        X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])                         #(1, 1, 160, 192, 224)
        F_X_Y = F.affine_grid(affine_matrix, X_label.shape, align_corners=True)                        #(1, 160, 192, 224, 3)
                
        X_Y_label = F.grid_sample(X_label, F_X_Y, mode="nearest", align_corners=True).cpu().numpy()        #(1, 1, 160, 192, 224)
        
        X_Y_label_DICE = X_Y_label[0, 0, :, :, :]                  #(160, 192, 224)
        Y_label = fixed_label.data.cpu().numpy()[0, 0, :, :, :]                              #(160, 192, 224)
        
        X_brain_label = (X_Y > 0).float().cpu().numpy()[0, 0, :, :, :]    
        # brain mask
        Y_brain_label = (fixed_img > 0).float().cpu().numpy()[0, 0, :, :, :]
        brain_dice = dice(np.floor(X_brain_label), np.floor(Y_brain_label))
        print("Brain dice score: ", brain_dice)
 
        dice_score = dice(np.floor(X_Y_label_DICE), np.floor(Y_label))
        print("Dice score: ", dice_score)
        
        X_Y_label_tensor = torch.from_numpy(X_Y_label).float().to(device)           #(1, 1, 160, 192, 224)
        
        X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]                      #(160, 192, 224)
        F_X_Y_cpu = X_Y_label_tensor.data.cpu().numpy()[0, 0, :, :, :]                 #(160, 192, 224)
        save_img(X_Y_cpu, f"{savepath}\\warpeed_image_ncc_{moving_base}", header=header, affine=affine)
        save_img(F_X_Y_cpu, f"{savepath}\\warpe_label_ncc_{moving_base}", header=label_header, affine=label_affine)

print("Result saved to :", savepath)