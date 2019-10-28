use_gradient_accumulating = True
path = '../input/understanding_cloud_organization'
num_workers = 0
bs = 1
num_epochs = 12
logdir = "./logs/segmentation"
if use_gradient_accumulating:
  output_logdir = "../input/cloud-master-output"
else:
  output_logdir = "../input/cloud-master-output-o/logs/segmentation"
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
original_height = 1400
original_width = 2100
input_height = 320
input_width = 640
inference_height = 350
inference_width = 525
