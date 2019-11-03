use_gradient_accumulating = True
path = '../input/understanding_cloud_organization'
num_workers = 0
if use_gradient_accumulating:
  bs = 1
else:
  bs = 16
num_epochs = 10
logdir = "./logs/segmentation"
output_logdir_list = ["../input/cloud-master-output/logs/segmentation",
                      "../input/cloud-master-output-o/logs/segmentation",
                      "../input/cloud-master-output-c/logs/segmentation"]
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
original_height = 1400
original_width = 2100
input_height = 320
input_width = 640
inference_height = 350
inference_width = 525
