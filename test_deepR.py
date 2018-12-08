import os,time,scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from model import SeeInDark
import config as cfg

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


def main():
   #get test IDs
   test_fns = glob.glob(cfg.gt_dir + '/0*.ARW')
   test_ids = []
   for i in range(len(test_fns)):
       _, test_fn = os.path.split(test_fns[i])
       test_ids.append(int(test_fn[0:5]))
   
   model = SeeInDark()
   model.load_state_dict(torch.load( cfg.pretrained_reconstructor))
   model = model.cuda()
   if not os.path.isdir(cfg.result_dir):
       os.makedirs(cfg.result_dir)
   
   for test_id in test_ids:
       #test the first image in each sequence
       in_files = glob.glob(cfg.input_dir + '%05d_00*.ARW'%test_id)
       for k in range(len(in_files)):
           in_path = in_files[k]
           _, in_fn = os.path.split(in_path)
           print(in_fn)
           gt_files = glob.glob(cfg.gt_dir + '%05d_00*.ARW'%test_id) 
           gt_path = gt_files[0]
           _, gt_fn = os.path.split(gt_path)
           in_exposure =  float(in_fn[9:-5])
           gt_exposure =  float(gt_fn[9:-5])
           ratio = min(gt_exposure/in_exposure,300)
   
           raw = rawpy.imread(in_path)
           im = raw.raw_image_visible.astype(np.float32) 
           im = im[:1024, :1024]
           input_full = np.expand_dims(pack_raw(im),axis=0) *ratio
   
           im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
           im = im[:1024, :1024]
           scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)	
   
           gt_raw = rawpy.imread(gt_path)
           im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
           im = im[:1024, :1024]
           gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)
   
           input_full = np.minimum(input_full,1.0)
   
           in_img = torch.from_numpy(input_full).permute(0,3,1,2).cuda()
           out_img = model(in_img)
           output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
   
           output = np.minimum(np.maximum(output,0),1)
    
           output = output[0,:,:,:]
           gt_full = gt_full[0,:,:,:]
           scale_full = scale_full[0,:,:,:]
           origin_full = scale_full
           scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
           
           scipy.misc.toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(cfg.result_dir + '%5d_00_%d_ori.png'%(test_id,ratio))
           scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(cfg.result_dir + '%5d_00_%d_out.png'%(test_id,ratio))
           scipy.misc.toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(cfg.result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
           scipy.misc.toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(cfg.result_dir + '%5d_00_%d_gt.png'%(test_id,ratio))
        
if __name__=='__main__':
   main()