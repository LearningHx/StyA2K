from email.policy import default
import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
from models.networks import *


class A2KModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', default='./checkpoints/vgg_normalised.pth', help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3', default=True,
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer', default=True,
                            help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=0., help='weight for L2 content loss')
            parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_no_param_A2K', type=float, default=3.,
                                help='weight for attention weighted style loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.visual_names = [ 'c','s','cs']
        self.model_names = ['decoder', 'transform']
        parameters = []
        self.max_sample = 64 * 64
        if opt.skip_connection_3:
            A2K = networks.A2K(in_planes=256,max_sample=self.max_sample)
            self.net_A2K = networks.init_net(A2K, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('A2K')
            parameters.append(self.net_A2K.parameters())
        
        channels = 512

        transform = networks.Transform(in_planes=512)
        decoder = networks.Decoder(opt.skip_connection_3)
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transform = networks.init_net(transform, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transform.parameters())
        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        if self.isTrain:
            self.loss_names = ['global', 'A2K', 'AdaA2K']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_A2K = torch.tensor(0., device=self.device)
            self.loss_AdaA2K = torch.tensor(0., device=self.device)
            self.sm = nn.Softmax(dim=-1)

    def set_input(self, input_dict,iter):
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']
        self.iter = iter
    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):
        self.c_feats = self.encode_with_intermediate(self.c)
        self.s_feats = self.encode_with_intermediate(self.s)
        c_A2K_feat_3 = self.net_A2K(self.c_feats[2], self.s_feats[2], layer='3',seed=self.seed)
        cs = self.net_transform(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4], self.seed)
        self.cs = self.net_decoder(cs, c_A2K_feat_3)

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(3, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(4):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
        
                self.loss_global += self.criterionMSE(stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_A2K = torch.tensor(0., device=self.device)
        self.loss_AdaA2K = torch.tensor(0., device=self.device)
        if self.opt.lambda_no_param_A2K > 0:
            for i in range(1,5):           
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                b,c,h,w = self.s_feats[i].shape
                if i == 2 or i==3:
                    region = 8
                    stride = 8
                elif i == 4:
                    region = 4
                    stride = 4
                elif i == 1:
                    region = 16
                    stride = 16
                head = 8
                s_value = self.s_feats[i]
                q_global = block(c_key,region,stride)
                v_global = block(s_value,region,stride)
                b,c1,n,r = q_global.shape
                b,c2,n,r = v_global.shape
                q = q_global.view(b,head,c1//head,n,r)
                k = block(s_key,region,stride).view(b,head,c1//head,n,r)
                v = block(s_value,region,stride).view(b,head,c2//head,n,r)
                
                DA_k_centers = torch.mean(k,dim=-1)
                DA_v_centers = torch.mean(v,dim=-1)
                dis =  torch.einsum("bhcx,bhcxy->bhxy",DA_k_centers,k)
                sim = torch.sigmoid(dis)
                DA_k_agg = (torch.einsum("bhxy,bhcxy->bhcx",sim,k) + DA_k_centers)/(r+1)
                DA_v_agg = (torch.einsum("bhxy,bhcxy->bhcx",sim,v) + DA_v_centers)/(r+1)
    
                logits = torch.einsum("bhcxy,bhcz->bhyxz",q,DA_k_agg)
                scores =  self.sm(logits)                                      #global
                DA_mean = torch.einsum("bhyxz,bhcz->bhcxy",scores,DA_v_agg)
            
                DA_mean_unblock =unblock(DA_mean.contiguous().view(b,c2,n,r),region,stride,h)

                PA1_logits = torch.einsum("bhcxy,bhczy->bhxz",q,k)
                index1 = torch.argmax(PA1_logits,dim = -1).view(b,head,1,n,1).expand_as(k) 
                index2 = torch.argmax(PA1_logits,dim = -1).view(b,head,1,n,1).expand_as(v)
                k_reshuffle = torch.gather(k,-2,index1)
                v_reshuffle = torch.gather(v,-2,index2)
                logits2 = torch.einsum("bhcxy,bhcxz->bhxyz",q,k_reshuffle)
                scores2 = self.sm(logits2)                                     #local
                PA_mean = torch.einsum("bhxyz,bhcxz->bhcxy",scores2,v_reshuffle)
                PA_mean_unblock =unblock(PA_mean.contiguous().view(b,c2,n,r),region,stride,h)


                O_mean = (DA_mean_unblock+PA_mean_unblock)
                self.loss_A2K += self.criterionMSE((stylized_feats[i]),(O_mean+self.c_feats[i]))
                
    def compute_losses(self):
        stylized_feats = self.encode_with_intermediate(self.cs)
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_A2K = self.loss_A2K * self.opt.lambda_no_param_A2K
        self.loss_global = self.loss_global * self.opt.lambda_global
        
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss =  self.loss_global + self.loss_A2K + self.loss_AdaA2K
        loss.backward()
        self.optimizer_g.step()

