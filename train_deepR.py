import os,time,scipy.io
import cv2
import numpy as np
import rawpy
import glob
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import SeeInDark
from yolo.deepR_models import *
from yolo.utils.utils import *
import config as cfg


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()
    
def detector_loss(out_img, gt_img, detector):
     '''calculate the loss for detector '''
     gt_img = Variable(gt_img.type(torch.cuda.FloatTensor))
     with torch.no_grad():
         gt_outputs = detector(gt_img)      
         gt_outputs = non_max_suppression(gt_outputs, 80, conf_thres=0.5, nms_thres=0.45) # a list
     #gt_output: (x1, y1, x2, y2, object_conf, class_score, class_pred)
           
     if gt_outputs[0] is None: 
         return torch.tensor(0.).cuda()
     print('gt_outputs', gt_outputs[0])
         
     # Get predicted boxes, confidence scores and labels
     pred_boxes = gt_outputs[0][:, :4] # size:(#cls, 4) 
     pred_labels = gt_outputs[0][:, -1]
     
     target = []
     for i, box in enumerate(pred_boxes):
        _pred_box = box.cpu().numpy() #.reshape((1,4)).tolist()
        pred_label = pred_labels[i].cpu().numpy().reshape((1,1)).tolist()  
        
        # reformat to x1, y1, x2, y2 and rescale to image dimensions
        pred_box =  np.empty_like(_pred_box)
        dw, dh = 1./cfg.image_size, 1./cfg.image_size
        pred_box[0] = abs(_pred_box[0] +_pred_box[1]) /2.0 #x 
        pred_box[1] = abs(_pred_box[2] + _pred_box[3])/2.0 #y
        pred_box[2] = abs(_pred_box[1] - _pred_box[0]) #w   
        pred_box[3] = abs(_pred_box[3] - _pred_box[2]) #h 
        pred_box[0] = pred_box[0]*dw
        pred_box[1] = pred_box[1]*dw
        pred_box[2] = pred_box[2]*dh
        pred_box[3] = pred_box[3]*dh
        pred_box = pred_box.reshape((1,4)).tolist()     
        one_target = pred_label[0] + pred_box[0]     
       
        one_target = np.array(one_target).reshape((1,5))
        one_target = torch.from_numpy(one_target[0]).float() # size(1,5)
        target.append(one_target)
       
     target = torch.stack(target) # a list of targets
     
     targets = [target]
     targets = torch.stack(targets)
     print('target', targets)
     print(targets.size())
     target = Variable(torch.FloatTensor(target).cuda(), requires_grad=False)
     loss = detector(out_img, target)
     
     return loss
     
def main():         
   #get train and test IDs
   train_fns = glob.glob(cfg.gt_dir + '0*.ARW')
   train_ids = []
   for i in range(len(train_fns)):
       _, train_fn = os.path.split(train_fns[i])
       train_ids.append(int(train_fn[0:5]))
   
   
   ps = cfg.ps #patch size for training
   save_freq = cfg.save_freq
   
   DEBUG = 0
   if DEBUG == 1:
       save_freq = 100
       train_ids = train_ids[0:5]   
   
   
   #Raw data takes long time to load. Keep them in memory after loaded.
   gt_images=[None]*6000
   input_images = {}
   input_images['300'] = [None]*len(train_ids)
   input_images['250'] = [None]*len(train_ids)
   input_images['100'] = [None]*len(train_ids)
   
   g_loss = np.zeros((5000,1))
   
   allfolders = glob.glob('./result/*0')
   
   for folder in allfolders:
       lastepoch = np.maximum(cfg.lastepoch, int(folder[-4:]))
   
   
   # create model
   model = SeeInDark()
   model.load_state_dict(torch.load(cfg.pretrained_path ))
   model.cuda()
   print("Model loaded from %s" % (cfg.pretrained_path) )
   print("now we are using %d gpus" %torch.cuda.device_count())
     
   YOLO = Darknet("yolo/config/yolov3.cfg", img_size=416)
   YOLO.load_weights(cfg.yolo_weights)
   detector = YOLO.cuda()
   print('------- loaded YOLO -------------')
   
   mse_loss = nn.MSELoss(size_average=True).cuda()  # Coordinate loss
   bce_loss = nn.BCELoss(size_average=True).cuda() # Confidence loss
   ce_loss = nn.CrossEntropyLoss().cuda() # Class loss
                 
   opt = optim.Adam(model.parameters(), lr = cfg.learning_rate)
   
   for epoch in range(cfg.lastepoch, cfg.epoches):
       if os.path.isdir("result/%04d"%epoch):
           continue    
       cnt=0
       if epoch > 2000:
           for g in opt.param_groups:
               g['lr'] = 1e-5
     
   
       for ind in np.random.permutation(len(train_ids)):
           # get the path from image id
           train_id = train_ids[ind]
           
           in_files = glob.glob(cfg.input_dir + '%05d_00*.ARW'%train_id)
           in_path = in_files[np.random.random_integers(0,len(in_files)-1)]
           _, in_fn = os.path.split(in_path)
   
           gt_files = glob.glob(cfg.gt_dir + '%05d_00*.ARW'%train_id)
           gt_path = gt_files[0]
           
           _, gt_fn = os.path.split(gt_path)
           in_exposure =  float(in_fn[9:-5])
           gt_exposure =  float(gt_fn[9:-5])
           ratio = min(gt_exposure/in_exposure,300)
             
           st=time.time()
           cnt+=1
   
           if input_images[str(ratio)[0:3]][ind] is None:
               # read dark images as input
               raw = rawpy.imread(in_path)            
               det_im = cv2.resize(pack_raw(raw), (512,512)) 
               det_im = np.expand_dims(det_im, axis=0)
               det_im = toTensor(det_im)
               input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio
               
               # read bright images as ground truth
               gt_raw = rawpy.imread(gt_path)
               gt_recons_im = cv2.resize(pack_raw(gt_raw), (1024,1024))
               gt_recons_im = np.expand_dims(gt_recons_im, axis = 0)
               gt_recons_im = toTensor(gt_recons_im)
              
           model.zero_grad()
           
           # reconstruct dark images to bright
           out_img = model(det_im) #[1, 3, 1024, 1024]
   
           gt_png = str(cfg.images_dir + 'train/%5d_00_%d_gt.png' % (train_id, ratio))
           gt_det_im = cv2.imread(gt_png)
           if gt_det_im is None:
              continue 
           gt_det_im = cv2.resize(gt_det_im, (1024,1024))      
           gt_det_im = np.expand_dims(gt_det_im, axis = 0)
           gt_det_im = toTensor(gt_det_im)
   
           # loss for reconstructor
           loss_r = reduce_mean(out_img, gt_det_im)
           
           # loss for detecotr
           loss_d = detector_loss(out_img, gt_det_im, detector)
                
           loss = loss_r +0.4*loss_d
           
           loss.backward()
   
           opt.step()
           g_loss[ind]=loss.data
   
           print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))
           
           if epoch%save_freq==0:
               if not os.path.isdir(cfg.result_dir + '%04d'%epoch):
                   os.makedirs(cfg.result_dir + '%04d'%epoch)
               torch.save(model.state_dict(), cfg.model_dir+'checkpoint_sony_e%04d.pth'%epoch)

if __name__=='__main__':
   main()