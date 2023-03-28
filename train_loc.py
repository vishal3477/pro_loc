from torchvision import datasets, models, transforms
#from model import *
import os
import torch
from torch import cdist
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
from model import *
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics
import cv2

import argparse
from functools import partial
import json
import traceback
import torch.nn.functional as F
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor
from vit_pytorch.recorder import Recorder
parser.add_argument('--model_path', default="/loc_model.pickle", help='pretrained model')


from torch.utils.data import Dataset, DataLoader


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

#import data
from STGAN import models
import data
import os
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--data_train',default='/mnt/scratch/asnanivi/man_gan_data',help='root directory for training data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--savedir', default='/mnt/scratch/asnanivi/runs')
parser.add_argument('--model_dir', default='./models')
parser.add_argument('--image_size', default=128, type=int, help='set size')
parser.add_argument('--template_strength', default=0.1, type=float, help='set size')
parser.add_argument('--resume', default=False, type=float, help='set size')





class encoder(nn.Module):
    def __init__(self, num_layers=10, num_features=64, out_num=1):
        super(encoder, self).__init__()

        layers_0 = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]

        layers_1=[]
        layers_2=[]
        layers_3=[]
        layers_4=[]
        for i in range(4):
            layers_1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                      nn.ReLU(inplace=True)))

        for i in range(3):
            layers_2.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        for i in range(3):
            layers_3.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))

        for i in range(3):
            layers_4.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))

        self.layers_0 = nn.Sequential(*layers_0)
        self.layers_1 = nn.Sequential(*layers_1)
        self.layers_2 = nn.Sequential(*layers_2)
        self.layers_3 = nn.Sequential(*layers_3)

        self.layers_5=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True))
        self.layers_6=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True))





    def forward(self, inputs):
        output = self.layers_0(inputs)
        output = self.layers_1(output)
        output_1 = self.layers_2(output)
        output_2 = self.layers_3(output)

        output_1 = self.layers_5(output_1)
        output_2 = self.layers_6(output_2)


        return output_1

    

class encoder1(nn.Module):
    def __init__(self, num_layers=6, num_features=64, out_num=2):
        super(encoder1, self).__init__()
        
        layers = [nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)
        self.layers2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers4=nn.Sequential(nn.Linear(65536, 512), nn.ReLU(), nn.Linear(512, 256), 
                              nn.ReLU(), nn.Linear(256, out_num), nn.Sigmoid())

    

    def forward(self, inputs):
        output1 = self.layers(inputs)
        output1 = self.layers2(output1)
        output1 = self.layers3(output1)
        output2 = output1.reshape(output1.size(0), -1)
        output2 = self.layers4(output2)
        
        return output2

    
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    #print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return real, imag


class vector_var(nn.Module):
    def __init__(self , size, set_size):
        super(vector_var, self).__init__()
        A = torch.rand(set_size,size,size, device='cpu')
        self.A = nn.Parameter(A)
        
    def forward(self):
        return self.A

opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)

size=opt.image_size
b_s=opt.batch_size
m=opt.template_strength

with open('./STGAN/output/%s/setting.txt' % 128) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']

label = args['label']
use_stu = args['use_stu']
stu_dim = args['stu_dim']
stu_layers = args['stu_layers']
stu_inject_layers = args['stu_inject_layers']
stu_kernel_size = args['stu_kernel_size']
stu_norm = args['stu_norm']
stu_state = args['stu_state']
multi_inputs = args['multi_inputs']
rec_loss_weight = args['rec_loss_weight']
one_more_conv = args['one_more_conv']

img = None
print('Using selected images:', img)

gpu = 'all'
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

#### testing
# multiple attributes
test_atts = ["Bald"] 
test_ints = 1.0

multi_atts = test_atts is not None
if test_atts is not None and test_ints is None:
    test_ints = [1 for i in range(len(test_atts))]
# single attribute
test_int = 1.0
# slide attribute
test_slide = False
n_slide = 10
test_att = None
test_int_min = -1.0
test_int_max = 1.0

thres_int = args['thres_int']
# others
use_cropped_img = args['use_cropped_img']
experiment_name = 128


device=torch.device("cuda:0")
torch.backends.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

sig = str(datetime.datetime.now())

print(sig)
    
train_path=opt.data_train
test_path=opt.data_test
save_dir=opt.savedir
os.makedirs('%s/logs/%s' % (save_dir, sig), exist_ok=True)
os.makedirs('%s/result_1/%s' % (save_dir, sig), exist_ok=True)

best_acc = 0
start_epoch = 1


set_size=1


encoder_model=encoder().to(device)  
optimizer_1 = torch.optim.Adam(encoder_model.parameters(), lr=0.00001)

signal_init=vector_var(size, set_size).cuda()
optimizer_2 = torch.optim.Adam(signal_init.parameters(), lr=0.000001)

signal=signal_init()

transformer=ViT(image_size = 128,patch_size = 8,num_classes = 256,dim = 64,depth = 6,heads = 4,mlp_dim = 512,dropout = 0.1,emb_dropout = 0.1).to(device) 

print(transformer)

lr = 0.000001
betas = (0.9, 0.999)
weight_decay = 0.5e-4
eps = 1e-8
    
#optimizer_3 = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
optimizer_3 = torch.optim.Adam(transformer.parameters(), lr=lr)

class_model=encoder1().to(device)  
optimizer_4 = torch.optim.Adam(class_model.parameters(), lr=opt.lr) 



sess = tl.session()
te_data = data.Celeba(train_path, atts, img_size, opt.batch_size, part='train', sess=sess, crop=not use_cropped_img, im_no=img)
# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
if use_stu:
    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                         test_label, is_training=False), test_label, is_training=False)
else:
    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './STGAN/output/%s/checkpoints' % experiment_name
tl.load_checkpoint(ckpt_dir, sess)



#filter_high=CannyFilter().cuda()


l2=torch.nn.MSELoss().to(device)
l_c=torch.nn.CrossEntropyLoss().to(device)
l_pair=torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
cos = nn.CosineSimilarity(dim=1, eps=1e-4)
cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)


state = {
    
    'state_dict_encoder':encoder_model.state_dict(),
    'optimizer_1': optimizer_1.state_dict(),
    'state_dict_signal':signal_init.state_dict(),
    'optimizer_2': optimizer_2.state_dict(),
    'state_dict_transformer':transformer.state_dict(),
    'optimizer_3': optimizer_3.state_dict(), 
    'state_dict_class':class_model.state_dict(),
    'optimizer_4': optimizer_4.state_dict()
    
}

if opt.resume:
    state1 = torch.load(opt.model_path)
    encoder_model.load_state_dict(state1['state_dict_encoder'])
    optimizer_1.load_state_dict(state1['optimizer_1'])
    signal_init.load_state_dict(state1['state_dict_signal'])
    optimizer_2.load_state_dict(state1['optimizer_2'])
    


def norm(tensor_map):
    tensor_map_AA=tensor_map.clone()
    tensor_map_AA = tensor_map_AA.view(tensor_map.size(0), -1)
    tensor_map_AA -= tensor_map_AA.min(1, keepdim=True)[0]
    tensor_map_AA /= (tensor_map_AA.max(1, keepdim=True)[0]-tensor_map_AA.min(1, keepdim=True)[0])
    tensor_map_AA = tensor_map_AA.view(tensor_map.shape)
    #tensor_map_AA[torch.isnan(tensor_map_AA)]=0
    
    return tensor_map_AA

def train(input_image, input_with_signal, gen_img_with_signal,signal_est,signal_set, transformer, encoder_model):
    
    encoder_model.train()
    transformer.train()
    class_model.train()
    
    transformer=Extractor(transformer)
    
    thirdPart_fft_1=torch.view_as_real(torch.fft.fft2(signal_est.clone()))
    thirdPart_fft_2=thirdPart_fft_1.clone()
    #print(thirdPart_fft_1.shape)
    thirdPart_fft_2[:,0,:,:],thirdPart_fft_2[:,0,:,:]=fftshift(thirdPart_fft_1[:,0,:,:],thirdPart_fft_1[:,1,:,:])
    signal_est_fs_shift=torch.sqrt(thirdPart_fft_2[:,0,:,:]**2+thirdPart_fft_2[:,1,:,:]**2+1e-10)
    
    
    
    n=50
    (_,w,h)=signal_est_fs_shift.shape
    half_w, half_h = int(w/2), int(h/2)

    signal_est_fs_low_freq=signal_est_fs_shift[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1].clone()
    target_zero = torch.zeros(signal_est_fs_low_freq.shape, dtype=torch.float32).to(device) 
    
    gt= gen_img_with_signal.type(torch.cuda.FloatTensor)-input_image.type(torch.cuda.FloatTensor)
    gt_gray=0.299*gt[:,0,:,:].clone()+0.587*gt[:,1,:,:].clone()+0.114*gt[:,2,:,:].clone()
    
    man_map_real, signal_real=encoder_model(input_with_signal)

    man_map_fake, signal_fake=encoder_model(gen_img_with_signal.type(torch.cuda.FloatTensor) )
    
    _, feat_real = transformer(input_with_signal.type(torch.cuda.FloatTensor))
    
    feat_trans_real=feat_real[:,1:,:]
    
    _, feat_fake = transformer(gen_img_with_signal.type(torch.cuda.FloatTensor))
    
    feat_trans_fake=feat_fake[:,1:,:]
    
    
    feat_trans_real_reshape=feat_trans_real.reshape(feat_trans_real.shape[0],256,8,8)
    feat_trans_fake_reshape=feat_trans_fake.reshape(feat_trans_fake.shape[0],256,8,8)
    
    trans_map_real=feat_trans_real.reshape(input_with_signal.shape[0],128,128)
    trans_map_fake=feat_trans_fake.reshape(input_with_signal.shape[0],128,128)
    
    
    transformer = transformer.eject()
    
    comb_maps=torch.cat((man_map_real, man_map_fake), dim=0)
    
    gt_class=torch.zeros((comb_maps.shape[0]), dtype=torch.long).to(device) 
    gt_class[int(comb_maps.shape[0]/2):]=1
    
    pred_class=class_model(comb_maps)
    
    comb_maps_trans=torch.cat((trans_map_real, trans_map_fake), dim=0)
    
    
    pred_class_trans=class_model(comb_maps_trans.unsqueeze(1))
    signal_real_AA=norm(signal_real)
    signal_fake_AA=norm(signal_fake)
    man_map_real_AA=norm(man_map_real)
    man_map_fake_AA=norm(man_map_fake)
    trans_map_real_AA=norm(trans_map_real)
    trans_map_fake_AA=norm(trans_map_fake)
    
    
    gt_AA=norm(gt_gray)
    signal_AA=norm(signal)
    
    
    zero=torch.zeros([1,input_with_signal.shape[2],input_with_signal.shape[3]], dtype=torch.float32).to(device) 
    loss1= 100*l2(signal,zero)
    
    loss2=(1. - cos(signal_est.reshape( signal_est.size(0), -1), signal_real_AA.reshape( signal_real_AA.size(0), -1)))
    loss2_tot=15*torch.sum(loss2)
    
    loss3=4*l2(signal_est_fs_low_freq, target_zero)
    
    loss4=cos(signal_est.reshape( signal_est.size(0), -1), signal_fake_AA.reshape( signal_fake_AA.size(0), -1))
    loss4_tot=20*torch.sum(loss4)
    
    
    
    
    
    zero=torch.zeros([input_with_signal.shape[0],1,input_with_signal.shape[2],input_with_signal.shape[3]], dtype=torch.float32).to(device) 
    loss5= 60*l2(man_map_fake,gt_gray)
    
    
    
    loss6=(1. - cos(gt_AA.reshape( gt_AA.size(0), -1), man_map_fake_AA.reshape( man_map_fake_AA.size(0), -1)))
    loss6_tot=25*torch.sum(loss6)
    
    
    loss7= 25*l2(trans_map_fake,gt_gray)
    
    
    
    loss8=(1. - cos(gt_AA.reshape( gt_AA.size(0), -1), trans_map_fake_AA.reshape( trans_map_fake_AA.size(0), -1)))
    loss8_tot=150*torch.sum(loss8)
    
    
    loss9= 50*l_c(pred_class,gt_class)
    
    loss10= 50*l_c(pred_class_trans,gt_class)
    
    loss11=0
    signal_set_norm=signal_set.clone()
    for i in range(signal_set.shape[0]):
        signal_set_norm[i]=(signal_set[i]-torch.min(signal_set[i]))/(torch.max(signal_set[i])-torch.min(signal_set[i]))
    #signal_set_norm[torch.isnan(signal_set_norm)]=0
    print(signal_set.shape)
    signal_set_norm_red=signal_set_norm.clone()
    for i in range(signal_set.shape[0]):
        for j in range(i):
            
            loss11+=cos1(signal_set_norm[i,:].reshape( -1), signal_set_norm[j,:].reshape( -1))
            #print(loss2)
    loss11_tot=5* loss11
    
    gt_AA_resize=F.interpolate(gt_AA.unsqueeze(1), size=256)
    man_map_fake_AA_resize=F.interpolate(man_map_fake_AA.unsqueeze(1), size=256)
    trans_map_fake_AA_resize=F.interpolate(trans_map_fake_AA.unsqueeze(1), size=256)
    
    
    loss12 =50*( 1 - ms_ssim( gt_AA_resize, man_map_fake_AA_resize, data_range=1, size_average=True, nonnegative_ssim=True))
    
    loss13=50*(1 - ms_ssim( gt_AA_resize, trans_map_fake_AA_resize, data_range=1, size_average=True, nonnegative_ssim=True))
    
    
    loss=loss1+loss2_tot+ loss3+ loss4_tot+ loss5+ loss6_tot+ loss7 + loss8_tot + loss9+ loss10+ loss11_tot+ loss12+ loss13
    print(loss, loss1,loss2_tot, loss3,loss4_tot, loss5, loss6_tot, loss7, loss8_tot, loss9, loss10, loss11_tot, loss12, loss13)
    

    
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    optimizer_3.zero_grad()
    optimizer_4.zero_grad()
    
    loss.backward()
    
    
    optimizer_1.step()
    optimizer_2.step()
    optimizer_3.step()
    optimizer_4.step()
    
    dist_fake_enc=torch.zeros([input_with_signal.shape[0],2], dtype=torch.float32).to(device)
    
    dist_fake_enc[:,0]=cos(gt_AA.reshape( gt_AA.size(0), -1), man_map_fake_AA.reshape(man_map_fake_AA.size(0), -1))
    dist_fake_enc[:,1]=cos(gt_AA.reshape( gt_AA.size(0), -1), man_map_real_AA.reshape(man_map_real_AA.size(0), -1))
    
    dist_fake_trans=torch.zeros([input_with_signal.shape[0],2], dtype=torch.float32).to(device)
    
    dist_fake_trans[:,0]=cos(gt_AA.reshape( gt_AA.size(0), -1), trans_map_fake_AA.reshape(man_map_fake_AA.size(0), -1))
    dist_fake_trans[:,1]=cos(gt_AA.reshape( gt_AA.size(0), -1), trans_map_real_AA.reshape(man_map_real_AA.size(0), -1))
    
    dist_fake=torch.zeros([b_s,set_size,2], dtype=torch.float32).to(device)
    for i in range(set_size):
        dist_fake[:,i,0]=cos(signal_set[i,:].reshape( -1).unsqueeze(0), signal_fake_AA.reshape(signal_fake.size(0), -1))
        dist_fake[:,i,1]=cos(signal_set[i,:].reshape( -1).unsqueeze(0), signal_real_AA.reshape(signal_fake.size(0), -1))
    
    
    return  signal,input_with_signal, man_map_real,gen_img_with_signal,man_map_fake, gt, dist_fake_enc, dist_fake_trans, pred_class,gt_class, dist_fake






epochs=40

count=0
flag=0
    
for idx, batch in enumerate(te_data):

    signal_est1=signal.clone()
    
    signal_sel=torch.randint(0,signal_est1.shape[0],(batch[0].shape[0],))
    print(signal_sel)
    
    signal_est2=signal_est1[signal_sel.type(torch.cuda.LongTensor),:]
    print(signal_est1.shape, signal_est2.shape)
    signal_est_red=m*signal_est2.clone() 
    
    print(batch[0].shape)
    print(torch.min(torch.tensor(batch[0])), torch.max(torch.tensor(batch[0])))
    print(torch.min(signal_est_red), torch.max(signal_est_red))
    xa_sample_ipt = (torch.tensor(batch[0]).permute(0,3,1,2).type(torch.cuda.FloatTensor) + signal_est_red.unsqueeze(1))
    a_sample_ipt = batch[1]
    b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(n_slide if test_slide else 1)]

    for a in test_atts:
        i = atts.index(a)
        b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
        b_sample_ipt_list[-1] = data.Celeba.check_attribute_conflict(b_sample_ipt_list[-1], atts[i], atts)

    x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
    raw_a_sample_ipt = a_sample_ipt.copy()
    raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
    for i, b_sample_ipt in enumerate(b_sample_ipt_list):
        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
        if not test_slide:

            if i > 0:   # i == 0 is for reconstruction
                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int
    gen_img= torch.tensor(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt.cpu().detach().permute(0,2,3,1),
                                                               _b_sample: _b_sample_ipt,
                                                               raw_b_sample: raw_a_sample_ipt}))



    signal ,input_with_signal, signal_rec,gen_img_with_signal,signal_fake, gt, dist_fake_enc, dist_fake_trans, pred_class,gt_class, dist_fake=train(torch.tensor(batch[0]).permute(0,3,1,2).type(torch.cuda.FloatTensor), xa_sample_ipt,gen_img.permute(0,3,1,2).type(torch.cuda.FloatTensor), signal_est_red, signal_est1, transformer, encoder_model)

    
    
    if flag==0:
        all_dist_fake_enc=dist_fake_enc.detach()
        all_dist_fake_trans=dist_fake_trans.detach()
        all_dist_fake=dist_fake.detach()
        all_pred_class=pred_class.detach()
        all_gt_class=gt_class.detach()
        flag=1
    else:
        all_dist_fake_enc=torch.cat([all_dist_fake_enc,dist_fake_enc.detach()], dim=0)
        all_dist_fake_trans=torch.cat([all_dist_fake_trans,dist_fake_trans.detach()], dim=0)
        all_dist_fake=torch.cat([all_dist_fake,dist_fake.detach()], dim=0)
        all_pred_class=torch.cat([all_pred_class,pred_class.detach()], dim=0)
        all_gt_class=torch.cat([all_gt_class,gt_class.detach()], dim=0)
        
        
    print(count)
    if count%19000==0:
        
        print('Saving model for count=',count)
        torch.save(state, '%s/result_1/%s/%d_model.pickle' % (save_dir, sig, count))
        print("Save Model: {:d}".format(count))
        
    
    count+=b_s
