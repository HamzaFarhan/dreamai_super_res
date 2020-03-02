from dreamai.utils import *
from dreamai.model import *
from dreamai.dai_imports import*
from dreamai_super_res import dbpn_v1
from dreamai_super_res.dbpn_discriminator import Discriminator, FeatureExtractor, FeatureExtractorResnet

class SuperResDBPN(Network):
    def __init__(self,
                 model_name = 'dbpn',
                 model_type = 'super_res',
                 lr = 0.08,
                 num_channels = 3,
                 base_filter = 64,
                 feat = 256,
                 upscale_factor = 4, 
                 criterion = nn.L1Loss(),
                 img_mean = [0.485, 0.456, 0.406],
                 img_std = [0.229, 0.224, 0.225],
                 inter_mode = 'bicubic',
                 residual = False,
                 denorm = False,
                 optimizer_name = 'adam',
                 device = None,
                 best_validation_loss = None,
                 best_psnr = None,
                 best_model_file = 'best_dbpn_sgd.pth',
                 model_weights = None,
                 optim_weights = None
                 ):

        super().__init__(device=device)

        print(f'Super Resolution using DBPN.')

        self.set_inter_mode(inter_mode)
        self.set_scale(upscale_factor)
        self.set_residual(residual)
        self.set_denorm(denorm)
        self.model = dbpn_v1.Net(num_channels=num_channels, base_filter=base_filter,
                                 feat=feat, num_stages=10, scale_factor=upscale_factor)
        # print(self.model.state_dict().keys())
        if model_weights:
            self.model.load_state_dict(model_weights)
        modules = list(self.model.module.named_modules())
        for n,p in modules:
            if isinstance(p, nn.Conv2d):
                setattr(self.model.module, n, nn.utils.weight_norm(p))
        self.model.to(device)
        self.set_model_params(criterion = criterion,optimizer_name = optimizer_name,lr = lr,model_name = model_name,model_type = model_type,
                              best_validation_loss = best_validation_loss,best_model_file = best_model_file)
        if optim_weights:
            self.optim.load_state_dict(optim_weights)
        self.best_psnr = best_psnr
        self.img_mean = img_mean
        self.img_std = img_std

    def set_denorm(self,denorm=True):
        self.denorm = denorm

    def set_scale(self,scale):
        self.upscale_factor = scale

    def set_inter_mode(self,mode):
        self.inter_mode = mode

    def set_residual(self,res):
        self.residual = res

    def enlarge(self, x):
        return self.predict(x)

    def forward(self,x):
        # if self.inter_mode is not None:
        #     res = F.interpolate(x.clone().detach(), scale_factor=self.upscale_factor, mode=self.inter_mode)
        x = self.model(x)
        # if self.residual:
            # x += res
        # if self.denorm:
            # x = denorm_tensor(x, self.img_mean, self.img_std)
            # x[:, 0, :, :] = x[:, 0, :, :] * self.img_std[0] + self.img_mean[0]
            # x[:, 1, :, :] = x[:, 1, :, :] * self.img_std[1] + self.img_mean[1]
            # x[:, 2, :, :] = x[:, 2, :, :] * self.img_std[2] + self.img_mean[2]
        
        return x

    def compute_loss(self,outputs,labels):

        ret = {}
        ret['mse'] = F.mse_loss(outputs,labels)
        loss = self.criterion(outputs, labels)
        ret['overall_loss'] = loss
        return loss,ret
    
    def evaluate(self,dataloader, **kwargs):

        # res = self.residual
        # self.set_residual(False)
        running_loss = 0.
        running_psnr = 0.
        rmse_ = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in dataloader:
                img, hr_target, hr_resized = data_batch[0],data_batch[1],data_batch[2]
                img = img.to(self.device)
                hr_target = hr_target.to(self.device)
                hr_super_res = self.forward(img)
                _,loss_dict = self.compute_loss(hr_super_res,hr_target)
                torchvision.utils.save_image([
                                            #   denorm_tensor(hr_target.cpu()[0], self.img_mean, self.img_std),
                                              hr_target.cpu()[0],
                                              hr_resized[0],
                                              hr_super_res.cpu()[0]
                                              ],
                                              filename='current_sr_model_performance.png')
                running_psnr += 10 * math.log10(1 / loss_dict['mse'].item())
                running_loss += loss_dict['overall_loss'].item()
                rmse_ += rmse(hr_super_res,hr_target).cpu().numpy()
        # self.set_residual(res)
        self.train()
        ret = {}
        ret['final_loss'] = running_loss/len(dataloader)
        ret['psnr'] = running_psnr/len(dataloader)
        ret['final_rmse'] = rmse_/len(dataloader)
        return ret

    # def predict(self,inputs,actv = None):
    #     res = self.residual
    #     self.set_residual(False)
    #     self.eval()
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         inputs = inputs.to(self.device)
    #         outputs = self.forward(inputs)
    #     if actv is not None:
    #         return actv(outputs)
    #     self.set_residual(res)
    #     return outputs
