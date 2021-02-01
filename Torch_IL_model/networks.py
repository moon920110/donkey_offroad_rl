import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out + x
    
class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class SeperatedImpalaNetwork(nn.Module):
    def __init__(self, discrete, steering_n=1, throttle_n=1):
        super(SeperatedImpalaNetwork, self).__init__()

        feature_size = 13057

        self.conv_layer = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        if discrete:
            self.steering_layer = nn.Sequential(
                nn.Linear(in_features=128, out_features=steering_n),
                nn.Softmax()
            )
        else:
            self.steering_layer = nn.Sequential(
                nn.Linear(in_features=128, out_features=steering_n)
            )

        self.throttle_layer = nn.Linear(in_features=128, out_features=throttle_n)

    def forward(self, img, scalar, get_feature=False):
        feature = self.conv_layer(img)
        total_feature = torch.cat((feature, scalar), 1)
        output = self.fc_layer(total_feature)

        steering = self.steering_layer(output)
        throttle = self.throttle_layer(output)

        action = torch.cat((steering, throttle), -1)
        if get_feature:
            return action, feature
        else:
            return action

class ImpalaNetwork(nn.Module):
    def __init__(self, steering_n=1, throttle_n=1):
        super(ImpalaNetwork, self).__init__()

        feature_size = 13057

        self.conv_layer = nn.Sequential(
            ImpalaBlock(in_channels=3, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=64),
            nn.Flatten()
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=steering_n+throttle_n)
        )

    def forward(self, img, scalar, get_feature=False):
        feature = self.conv_layer(img)
        total_feature = torch.cat((feature, scalar), 1)
        output = self.fc_layer(total_feature)
        if get_feature:
            return output, feature
        else:
            return output


class LightNetwork(nn.Module):
    def __init__(self, steering_n=1, throttle_n=1):
        super(LightNetwork, self).__init__()

        feature_size = 958528

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Flatten()
        )

        self.speed_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=64),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=64, out_features=steering_n+throttle_n)
        )

    def forward(self, img, scalar, get_feature=False):
        feature = self.conv_layer(img)
        speed_feature = self.speed_layer(scalar)
        total_feature = torch.cat((feature, speed_feature), 1)
        output = self.fc_layer(total_feature)
        if get_feature:
            return output, feature
        else:
            return output
