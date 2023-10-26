import torch
import torch.nn as nn
# import torchvision.models as models
import sys
sys.path.append('src')
from resnet import ResNet
import torch.nn.functional as F
from torch.cuda.amp import autocast


class AutoConv2d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0):
        super(AutoConv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 在第一次运行时，根据输入的通道数创建卷积层
        in_channels = x.size(1)
        self.conv2d = nn.Conv2d(in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.conv2d.to(x.device)

        return self.conv2d(x)

class DifferenceFeatureExtractor(nn.Module):
    def __init__(self):
        super(DifferenceFeatureExtractor, self).__init__()
        resnet50 = ResNet([2, 2, 2, 2])

        self.layer1 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.stage1)
        self.layer2 = resnet50.stage2
        self.layer3 = resnet50.stage3
        self.layer4 = resnet50.stage4
        self.convs = nn.ModuleList([AutoConv2d(1, 1) for _ in range(4)]) # 创建列表来存储卷积层

    def extract_features(self, x, layer):
        return layer(x)

    def forward(self, image1, image2):
        input_size = image1.size()[2:] # 保存输入图像的尺寸，用于上采样
        diff_features = []
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            feat1 = self.extract_features(image1, layer)
            feat2 = self.extract_features(image2, layer)
            diff = torch.abs(feat1 - feat2)

            diff_global_avg_pool = F.adaptive_avg_pool2d(diff, (224, 224)) # 全局平均池化
            del diff
            #print(diff_global_avg_pool)
            diff_upsampled = F.interpolate(diff_global_avg_pool, size=input_size, mode='bilinear', align_corners=False) # 上采样到输入图像尺寸
            del diff_global_avg_pool
            diff_conv = self.convs[idx](diff_upsampled) # 使用存储在列表中的卷积层
            del diff_upsampled
            diff_features.append(diff_conv)
            del diff_conv
            image1, image2 = feat1, feat2
            del feat1,feat2
            torch.cuda.empty_cache()
        return torch.cat(diff_features, dim=1)


class CrossAttentionFusionModule(nn.Module):
    def __init__(self):
        super(CrossAttentionFusionModule, self).__init__()

        # 降采样
        self.downsample = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # 输出大小：(batch, 64, 64, 64)

        # 多头自注意力
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        # 升采样
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )  # 输出大小：(batch, 16, 256, 256)

        # 通道调整
        self.channel_adjust = nn.Conv2d(16, 3, kernel_size=1)  # 输出大小：(batch, 3, 256, 256)

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(2, 0, 3, 1).contiguous().view(64 * 64, -1, 64)  # 为自注意力调整形状
        x, _ = self.attention(x, x, x)
        x = x.view(64, x.size(1), 64, 64).permute(1, 3, 0, 2).contiguous()  # 恢复原始形状
        x = self.upsample(x)
        x = self.channel_adjust(x)
        return x

class InverseDiffusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps

        # 定义一个基本的卷积网络，用于从当前状态生成噪声
        self.transition_model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # beta为每个时间步的噪声系数，可以根据具体的扩散过程进行设置
        beta = torch.linspace(0.1, 0.2, self.num_timesteps)

        # 逆扩散过程：从噪声数据恢复原始数据
        for t in reversed(range(self.num_timesteps)):
            noise = self.transition_model(x) * torch.sqrt(beta[t])
            x = (x - noise) / (1 - beta[t])

        return x

class GlacierChangeDetection(nn.Module):
    def __init__(self):
        super(GlacierChangeDetection, self).__init__()
        self.diff_extractor = DifferenceFeatureExtractor()
        # self.cross_attention = CrossAttentionFusion(d_model=4, nhead=4) # 选择合适的d_model和nhead
        self.inverse_diffusion = InverseDiffusion(in_channels=4, hidden_channels=64, num_timesteps=10)

    def forward(self, image1, image2):
        diff_features = self.diff_extractor(image1, image2)
        prediction = self.inverse_diffusion(diff_features)
        prediction = F.softmax(prediction, dim=1) # 使用Softmax获取每个像素的类别概率
        return prediction

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kl = GlacierChangeDetection()
    kl.to(device)
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)
    x1 = x1.to(device)
    x2 = x2.to(device)
    with autocast():  # 使用自动混合精度
        X = kl(x1, x2)

    print(X.shape)
