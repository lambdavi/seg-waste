import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
from models.bisenet import BiSeNet
#from models.icnet import ICNet
from config import cfg
from loading_data import loading_data
from utils.utils import *
from timer import Timer
from utils.stream_metrics import StreamSegMetrics
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

"""
    if cfg.TASK == 'binary':
    train_metric = StreamSegMetrics(2, "train")
    val_metric = StreamSegMetrics(2, "val")
else:
    train_metric = StreamSegMetrics(cfg.DATA.NUM_CLASSES+1, "train")
    val_metric = StreamSegMetrics(cfg.DATA.NUM_CLASSES+1, "val")
"""

train_metric = StreamSegMetrics(cfg.DATA.NUM_CLASSES, "train")
val_metric = StreamSegMetrics(cfg.DATA.NUM_CLASSES, "val")

def main():
    # TODO Create a skeleton OOP
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch.backends.cudnn.benchmark = True
    if cfg.TASK == "binary":
        criterion = torch.nn.BCEWithLogitsLoss().to(device)# Binary Classification
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    net = []   
    
    if cfg.MODEL == "enet":
        if cfg.TRAIN.STAGE=='all':
            net = ENet(only_encode=False)
            if cfg.TRAIN.PRETRAINED_ENCODER != '':
                encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
                del encoder_weight['classifier.bias']
                del encoder_weight['classifier.weight']
                # pdb.set_trace()
                net.encoder.load_state_dict(encoder_weight)
        elif cfg.TRAIN.STAGE =='encoder':
            net = ENet(only_encode=True)
    elif cfg.MODEL == "bisenetv2":   
        net = BiSeNetV2(cfg.DATA.NUM_CLASSES, pretrained=True)
    else:
        net = BiSeNet(cfg.DATA.NUM_CLASSES, None, None, None) # get Bisenetv1

    net=net.to(device)
    
    if cfg.LOAD:
        """try:
            net.load_state_dict(torch.load("models/saved_models/best_model.pth"))
            print("Model loaded successfully.")
        except:
            print("Error in loading the model")
            exit()"""
        net.load_state_dict(torch.load("models/saved_models/best_model.pth"))
        net=net.to(device)

    else:
        _t = {'train time' : Timer(),'val time' : Timer()} 
        optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        reduction  = MeanReduction()
        scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
        validate(val_loader, net, criterion, optimizer, -1, restore_transform, device)
    
        print("Starting training..")
        for epoch in range(cfg.TRAIN.MAX_EPOCH):
            print(f"### \tEpoch {epoch+1}/{cfg.TRAIN.MAX_EPOCH}\t ###")
            _t['train time'].tic()
            print(" --- TRAINING --- ")
            train(train_loader, net, criterion, reduction, optimizer, epoch, device)
            _t['train time'].toc(average=False)
            #print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
            _t['val time'].tic()
            print(" --- VALIDATION --- ")
            validate(val_loader, net, criterion, optimizer, epoch, restore_transform, device)
            _t['val time'].toc(average=False)
            #print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))
    
    if cfg.PRED_PATH:
        predict(cfg.PRED_PATH, train_loader, net, device)
        
    if cfg.SAVE and not cfg.LOAD:
        torch.save(net.state_dict(), "models/saved_models/best_model.pth")

    
    
def update_metric(metric, outputs, labels):
        """
        Update the evaluation metric with the model outputs and labels.

        Args:
            `metric`: Metric object to be updated.\n
            `outputs`: Model outputs.\n
            `labels`: True labels.

        """
        _, prediction = outputs.max(dim=1)
        labels = labels.detach().cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

def print_results(metric):
        results = metric.get_results()
        print(f"\tMean IOU: {results['Mean IoU']}")
        print(f"\tClass IOU: {results['Class IoU']}")
        print(f"\tClass Accuracy: {results['Class Acc']}")


def train(train_loader, net, criterion, reduction, optimizer, epoch, device="cpu"):
    train_metric.reset()
    torch.cuda.empty_cache()
    for inputs, labels in tqdm(train_loader, ascii=True):
        inputs = Variable(inputs).to(device, dtype=torch.float32)
        labels = Variable(labels).to(device, dtype=torch.long)

        if cfg.MODEL == "enet":
            outputs = net(inputs)
        elif cfg.MODEL == "bisenetv2":
            outputs = net(inputs, test=False)[0]
        else:
            outputs = net(inputs)

        if cfg.TASK == "binary":
            loss = criterion(outputs, labels.unsqueeze(1).float())
        else:
            loss = reduction(criterion(outputs,labels),labels)

        #print(torch.unique(labels))
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        if cfg.TASK == "binary":
            out_metr = outputs.detach()
            out_metr[out_metr > 0.5] = 1
            out_metr[out_metr<=0.5] = 0    
            train_metric.update(labels.cpu().numpy(), out_metr.cpu().numpy())
        else:
            update_metric(train_metric, outputs, labels)
            #train_metric.update(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    
    print_results(train_metric)

def validate(val_loader, net, criterion, optimizer, epoch, restore, device):
    net.eval()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    val_metric.reset()
    with torch.no_grad():
        for vi, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            if cfg.MODEL == "enet":
                outputs = net(inputs)
            elif cfg.MODEL == "bisenetv2":
                outputs = net(inputs, test=False)[0]
            else:
                outputs = net(inputs)

            if cfg.TASK == "binary":
                #for binary classification
                outputs[outputs>0.5] = 1
                outputs[outputs<=0.5] = 0
                val_metric.update(labels.cpu().numpy(), outputs.cpu().numpy())
            else:
                update_metric(val_metric, outputs, labels)

        
        """if cfg.TASK == "binary":
            iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
        else:
            iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 4)"""

    #mean_iu = iou_/len(val_loader)   

    #print('\t[mean iu %.4f]' % (mean_iu)) 
    print_results(val_metric)

    net.train()

def predict2(image_path, train_loader, model, device):
    input_image = Image.open(image_path)

    # Apply necessary transformations
    transforms = train_loader.dataset.transform

    # Add batch dimension
    input_tensor = transforms(input_image).unsqueeze(0)  
    input_tensor = input_tensor.to(device)
    model.eval()
    # Perform inference
    with torch.no_grad():
        if cfg.MODEL == "enet":
            output = model(input_tensor)  # Get the output logits
        elif cfg.MODEL == "bisenetv2":
            output = model(input_tensor, test=False)[0]
        else:
            output = model(input_tensor)

    output = output.squeeze(0).cpu().numpy()
    print(output.shape)
    translator = {0: 255, 1: 50, 2: 100, 3: 150, 4: 200}
    predicted_labels = np.argmax(output, axis=0)
    fin = np.vectorize(translator.get)(predicted_labels)
    _, ax = plt.subplots()
    ax.imshow(fin)
    plt.savefig('class_img.png', bbox_inches='tight', dpi=300)




def predict(image_path, train_loader, model, device):
    """
    Handles the the prediction. Outputs an image in the root directory.
    Args: 
        `image_path`: path to the image to predict.
    """
        
    # Load and preprocess the input image
    input_image = Image.open(image_path)

    # find gt
    splits = image_path.split("/")
    gt_path = splits[0]+"/labels/"+splits[1]+"/"+splits[2]
    print(gt_path)
    input_gt = Image.open(gt_path)

    # Apply necessary transformations
    transforms = train_loader.dataset.transform
    target_transforms = train_loader.dataset.target_transform
    # Add batch dimension
    input_tensor = transforms(input_image).unsqueeze(0)  
    gt_tensor = target_transforms(input_gt).unsqueeze(0)  

    input_tensor = input_tensor.to(device)
    gt_tensor = gt_tensor.to(device)
    model.eval()
    # Perform inference
    with torch.no_grad():
        if cfg.MODEL == "enet":
            output = model(input_tensor)  # Get the output logits
        elif cfg.MODEL == "bisenetv2":
            output = model(input_tensor, test=True)[0]
        else:
            output = model(input_tensor)

    output = output.squeeze(0).cpu().numpy()
    print(output.shape)

    predicted_labels = np.argmax(output, axis=0)
    print(np.unique(predicted_labels))

    # Convert labels into masks
    gt_tensor[gt_tensor!=255] = 1
    gt_tensor[gt_tensor==255] = 0

    mask = gt_tensor

    #translator 
    predicted_labels    

    class_names = ["alluminum", "carton", "bottle", "nylon"]

    # Get colormap
    colormap = plt.cm.get_cmap('tab20', len(class_names))

    # Create the predicted image with colors
    predicted_image = Image.fromarray((colormap(predicted_labels) * 255 ).astype(np.uint8))    
    # Create a legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colormap(i)) for i in range(len(class_names))]

    # Create a figure and axes
    _, ax = plt.subplots()

    # Display the predicted image
    #ax.imshow(np.array(input_image))

    ax.imshow(predicted_image)
    
    ax.axis('off')

    # Create the legend outside the image
    legend = ax.legend(legend_elements, class_names, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the positioning and appearance of the legend
    legend.set_title('Legend')
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_facecolor('white')

    # Save the figure
    plt.savefig('class_img.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()








