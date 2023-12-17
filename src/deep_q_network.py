"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear((1 * 20 * 10) + 3, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)  # Additional layer
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, 32)  # Additional layer
        self.relu6 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(32, 1)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure that x has at least two dimensions
        x_image = x[:, :200].view(-1, 1, 20, 10)
        x_extra = x[:, 200:].view(-1, 3)  # Extract the last 2 elements

        # Process the image only if it is not None
        if x_image is not None:
            x_image = self.conv1(x_image)
            x_image = self.relu1(x_image)
            x_image = self.conv2(x_image)
            x_image = self.relu2(x_image)
            x_image = self.conv3(x_image)
            x_image = self.relu3(x_image)

            # Flatten the image output before concatenating with extra features
            x_image = x_image.view(x_image.size(0), -1)

        # Concatenate with extra features
        x_combined = torch.cat([x_image, x_extra], dim=1)

        x_combined = self.fc1(x_combined)
        x_combined = self.relu4(x_combined)
        x_combined = self.fc2(x_combined)
        x_combined = self.relu5(x_combined)
        x_combined = self.fc3(x_combined)
        x_combined = self.relu6(x_combined)
        x_combined = self.fc4(x_combined)

        return x_combined
