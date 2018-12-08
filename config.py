import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../../Learning-to-See-in-the-Dark/dataset/Sony/short/', help='path to dark images')
parser.add_argument('--gt_dir', type=str, default='../../Learning-to-See-in-the-Dark/dataset/Sony/long/', help='path to bright images')
parser.add_argument('--result_dir', type=str, default='../result_Sony_deepR/', help='path to results')
parser.add_argument('--model_dir', type=str, default='../saved_model/', help='path to pretrained models')
parser.add_argument('--yolo_weights', type=str, default="/home/hy128/pytorch-Learning-to-See-in-the-Dark/yolo/weights/yolov3.weights", help='path to pretrained detector weights')

opt = parser.parse_args()
print(opt)

input_dir = opt.input_dir
gt_dir = opt.gt_dir
result_dir = opt.result_dir
model_dir = opt.model_dir
yolo_weights = opt.yolo_weights

ps = 512 #patch size for training
save_freq = 200

image_size = 1024.

learning_rate = 1e-4

lastepoch = 4000
epoches = 10001
pretrained_path = "/home/hy128/pytorch-Learning-to-See-in-the-Dark/saved_model_backup/checkpoint_sony_e4000.pth"
images_dir = '/home/hy128/Learning-to-See-in-the-Dark/result_Sony/'

pretrained_reconstructor = '/home/hy128/pytorch-Learning-to-See-in-the-Dark/saved_model/checkpoint_sony_e4800.pth'

batch_size = 1
model_config_path = "yolo/config/yolov3.cfg"
data_config_path = "yolo/config/coco.data"

class_path = "yolo/data/coco.names"

iou_thres = 0.5
conf_thres = 0.5 #object confidence threshold
nms_thres = 0.45 #iou thresshold for non-maximum suppression
n_cpu = 0 #number of cpu threads to use during batch generation