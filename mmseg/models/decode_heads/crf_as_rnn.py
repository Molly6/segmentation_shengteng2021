"""
PyTorch implementation of CRFasRNN for semantic segmentation.
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Parameter


def make_onehot_kernel(kernel_size, index):
    """
    Make 2D one hot square kernel, i.e. h=w
    k[kernel_size, kernel_size] = 0 except k.view(-1)[index] = 1
    """
    kernel = torch.zeros(kernel_size, kernel_size)
    kernel.view(-1)[index] = 1
    return kernel.view(1, 1, kernel_size, kernel_size)


def make_spatial_kernel(kernel_size, bandwidth, isreshape=True):
    """
    Make 2D square smoothness kernel, i.e. h=w
    k = 1/bandwidth * exp(-(pj-pi)**2/(2*bandwidth**2))
    pj, pi = location of pixel
    """
    assert bandwidth > 0, 'bandwidth of kernel must be > 0'
    assert kernel_size % 2 != 0, 'kernel must be odd'
    p_end = (kernel_size - 1) // 2  #kernel center indices
    X = torch.linspace(-p_end, p_end,
                       steps=kernel_size).expand(kernel_size, kernel_size)
    Y = X.clone().t()
    kernel = torch.exp(-(X**2 + Y**2) / (2 * (bandwidth**2)))
    #! due to the require of paper: j#i, thus when j=i, kernel=0
    kernel[p_end, p_end] = 0
    if isreshape:
        return kernel.view(1, 1, kernel_size, kernel_size)
    return kernel


class GaussianMask(nn.Module):
    """
    Break down Gaussian kernel (2nd part of appearance kernel) into CNN
    kj = (I(j) - I(i))**2/2*bandwidth**2, j#i
    but compute all maps instead of 1 kernel
    """
    def __init__(self, in_channels, kernel_size, bandwidth, iskernel=True):
        super(GaussianMask, self).__init__()
        assert bandwidth > 0, 'bandwidth of kernel must be > 0'
        assert kernel_size % 2 != 0, 'kernel must be odd'
        self.bandwidth = bandwidth
        self.iskernel = iskernel
        self.n_kernels = kernel_size**2 - 1
        kernel_weight = self._make_kernel_weight(in_channels, kernel_size,
                                                 self.n_kernels)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, in_channels*self.n_kernels, kernel_size, \
            stride=1, padding=padding, groups=in_channels, bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(kernel_weight.view_as(self.conv.weight))

    def _make_kernel_weight(self, in_channels, kernel_size, n_kernels):
        #! Be carefull with contruct weight, otherwise, output will be mixed in unwanted order
        kernel_weight = torch.zeros(in_channels, n_kernels, kernel_size,
                                    kernel_size)
        for i in range(n_kernels):
            index = i if i < (n_kernels // 2) else i + 1
            kernel_i = make_onehot_kernel(kernel_size, index)
            kernel_weight[:, i, :] = kernel_i
        return kernel_weight

    def forward(self, X):
        #compute (I(j)-I(i))**2/(2*bandwidth**2)
        batch_size, in_channels, H, W = X.shape
        Xj = self.conv(X).view(batch_size, in_channels, self.n_kernels, H, W)
        if not self.iskernel:
            return Xj
        Xi = X.unsqueeze(dim=2)
        K = (Xj - Xi)**2 / (2 * (self.bandwidth**2))
        K = torch.exp(-K)
        return K  #size B*C*N_ker*H*W


class SpatialFilter(nn.Module):
    """
    Break down spatial filter (smoothest kernel) into CNN blocks
    refer: https://arxiv.org/pdf/1210.5644.pdf
    """
    def __init__(self, n_classes, kernel_size, theta_gamma):
        super(SpatialFilter, self).__init__()
        padding = kernel_size // 2
        kernel_weight = make_spatial_kernel(kernel_size, theta_gamma)
        self.conv = nn.Conv2d( \
            n_classes, n_classes, kernel_size, \
            stride=1, padding=padding, groups=n_classes, bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(kernel_weight)

    def forward(self, Q):
        Qtilde = self.conv(Q)  #size B*N_class*H*W
        norm_weight = self.conv(Q.new_ones(*Q.shape, requires_grad=False))
        Qtilde = Qtilde / norm_weight
        return Qtilde


class BilateralFilter(nn.Module):
    """
    Break down bilateral filter (appearance kernel) into CNN blocks
    remember that exp(-a-b) =exp(-a)*exp(b)
    """
    def __init__(self, in_channels, n_classes, kernel_size, theta_alpha,
                 theta_beta):
        super(BilateralFilter, self).__init__()
        #need 6 dims for later purpose
        kernel_weight = make_spatial_kernel(kernel_size,
                                            theta_alpha,
                                            isreshape=False)
        self.spatial_weight = Parameter( \
            kernel_weight[kernel_weight > 0].view(1, 1, 1, -1, 1, 1), requires_grad=False) #remove center
        self.gauss_mask_I = GaussianMask(in_channels, kernel_size, theta_beta)
        self.guass_mask_Q = GaussianMask(n_classes,
                                         kernel_size,
                                         1,
                                         iskernel=False)
        #TODO: set requires_grad of params to False

    def forward(self, Q, I):
        #make masks for filters
        Ij = self.gauss_mask_I(I)  #size B*C*N_ker*H*W
        Qj = self.guass_mask_Q(Q)  #size B*N_class*N_ker*H*W
        Qj = Ij.unsqueeze(dim=2) * Qj.unsqueeze(
            dim=1)  #size B*C*N_class*N_ker*H*W
        #multiply with spatial weight on N_ker dimension
        Qj = Qj * self.spatial_weight
        #sum over spatial weight dimension
        Qtilde = Qj.sum(dim=3)  ##size B*C*N_class*H*W, thus C=M in the paper
        #norm
        norm_weight = Ij * self.spatial_weight.squeeze(
            dim=2)  #size B*C*N_ker*H*W
        norm_weight = norm_weight.sum(dim=2)  #size B*C*H*W
        Qtilde = Qtilde / norm_weight.unsqueeze(dim=2)
        return Qtilde


class MessagePassing(nn.Module):
    """
    Combine bilateral filter (appearance filter)
    and spatial filter to make message passing
    """
    def __init__(self,
                 in_channels,
                 n_classes,
                 kernel_size=[3],
                 theta_alpha=[2.],
                 theta_beta=[2.],
                 theta_gamma=[2.]):
        super(MessagePassing, self).__init__()
        assert len(theta_alpha) == len(
            theta_beta), 'theta_alpha and theta_beta have different lengths'
        # self.bilateralfilter = BilateralFilter(in_channels, n_classes, kernel_size, theta_alpha, theta_beta)
        # self.spatialfilter = SpatialFilter(n_classes, kernel_size, theta_gamma)
        self.n_bilaterals, self.n_spatials = len(theta_alpha), len(theta_gamma)
        for i in range(self.n_bilaterals):
            self.add_module( \
                'bilateral{}'.format(i), \
                BilateralFilter(in_channels, n_classes, kernel_size[i], theta_alpha[i], theta_beta[i]))
        for i in range(self.n_spatials):
            self.add_module(
                'spatial{}'.format(i),
                SpatialFilter(n_classes, kernel_size[i], theta_gamma[i]))

    def _get_child(self, child_name):
        return getattr(self, child_name)

    def forward(self, Q, I):
        # bilateralQ = self.bilateralfilter(Q, I) #B*n_bilaterals*N_class*H*W
        # spatialQ = self.spatialfilter(Q) #B*N_class*H*W
        filteredQ = []
        for i in range(self.n_bilaterals):
            tmp_bilateral = self._get_child('bilateral{}'.format(i))(Q, I)
            filteredQ.append(tmp_bilateral)
        for i in range(self.n_spatials):
            tmp_spatial = self._get_child('spatial{}'.format(i))(Q)
            filteredQ.append(tmp_spatial.unsqueeze(dim=1))
        # B*(n_bilaterals+n_spatials)*N_class*H*W
        Qtilde = torch.cat(filteredQ, dim=1)
        return Qtilde


class CRFRNN(nn.Module):
    """ Break meanfields down as CNN and do iteration """
    def __init__(self,
                 n_iter,
                 in_channels,
                 n_classes,
                 kernel_size=[3, 3],
                 theta_alpha=[1.5, 2.5],
                 theta_beta=[1.5, 2.5],
                 theta_gamma=[1.5]):
        super(CRFRNN, self).__init__()
        self.n_iter = n_iter
        self.n_classes = n_classes
        n_filters = in_channels * len(theta_alpha) + len(theta_gamma)
        self.softmax = nn.Softmax2d()  #nn.Softmax(dim=1)
        self.messagepassing = MessagePassing(in_channels,
                                             n_classes,
                                             kernel_size=kernel_size,
                                             theta_alpha=theta_alpha,
                                             theta_beta=theta_beta,
                                             theta_gamma=theta_gamma)
        self.weightfiltering = Parameter(
            torch.rand(1, n_filters, n_classes, 1, 1))
        self.compatibilitytransf = nn.Conv2d( \
            n_classes, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self._weight_initial()
        self.train_step = 0

    def _weight_initial(self):
        init.kaiming_normal_(self.weightfiltering)
        init.kaiming_normal_(self.compatibilitytransf.weight)

    def forward(self, U, I):
        if self.training:
            if self.train_step<60000:
                self.n_iter = 1
            elif self.train_step<70000:
                self.n_iter = 2
            elif self.train_step<75000:
                self.n_iter = 3
            else:
                self.n_iter = 4
            self.train_step = self.train_step +1
        else:
            self.n_iter = 8

        Q = U
        for _ in range(self.n_iter):
            #normalize
            Q = self.softmax(Q)
            #message passing
            Q = self.messagepassing(Q, I)
            #weight filtering
            Q = Q * self.weightfiltering
            Q = Q.sum(dim=1)
            #compatibility transform
            #need to minus Q*weight because sum(mu_l'l * Q_l') with l'#l
            Q = self.compatibilitytransf(Q) \
                - Q * self.compatibilitytransf.weight.squeeze().diag().view(1, self.n_classes, 1, 1)
            #adding unary
            Q = U - Q
        return Q


def count_model_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    crf = CRFRNN(5, 3, 19)
    print('Total params num: {count_model_param(crf)}')