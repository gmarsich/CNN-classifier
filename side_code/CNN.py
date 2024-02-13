from torch import nn
import torch.nn.init as init


# CNN requested by Task 1
class CNN1(nn.Module):

    # A model will have an __init__() function, where it instantiates its layers

    def __init__(self):
        super(CNN1, self).__init__() # the constructor of the parent class (nn.Module) is called to initialize the model properly

        # Convolutional layer 1: in_channels=1 because we have a greyscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1) # from [1] we get the formula: output = ((input - kernel_size + 2*padding)/stride) + 1 => 62*62
        # ReLU activation after conv1
        self.relu1 = nn.ReLU() # output: 62*62
        # Max pooling layer 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 31*31 (from 62/2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1) # from [1] we know that output: 29*29
        # ReLU activation after conv2
        self.relu2 = nn.ReLU() # output: 29*29
        # Max pooling layer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 14*14 (from the test in dim_images.ipynb)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1) # from [1] we know that output: 12*12
        # ReLU activation after conv3  
        self.relu3 = nn.ReLU() # output: 12*12

        # Fully connected layer. 32: number of channels; 12, 12: height and width of the feature map
        self.fc = nn.Linear(32 * 12 * 12, 15)

        self.initialize_weights()


    def initialize_weights(self):
        for module in self.modules(): # self.modules() comes from nn.Module; to recursively iterate over all the modules
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01) # initial weights drawn from a Gaussian distribution having a mean of 0 and a standard deviation of 0.01
                init.constant_(module.bias, 0) # set the bias to 0


    # A model will have a forward() function
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(-1, 32 * 12 * 12)  # flatten the tensor before passing to fully connected layers (the size -1 is inferred from other dimensions)
        
        x = self.fc(x)

        return x




# CNN with batch normalization, different size of convolutional filters
class CNN2(nn.Module):

    # A model will have an __init__() function, where it instantiates its layers

    def __init__(self):
        super(CNN2, self).__init__() # the constructor of the parent class (nn.Module) is called to initialize the model properly.

        # Convolutional layer 1: in_channels=1 because we have a greyscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1) # from [1] we get the formula: output = ((input - kernel_size + 2*padding)/stride) + 1 => 62*62
        # batch normalization
        self.bn1 = nn.BatchNorm2d(8) # the parameter should be equal to the out_channels of the convolution
        # ReLU activation after conv1
        self.relu1 = nn.ReLU() # output: 62*62
        # Max pooling layer 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 31*31 (from 62/2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1) # from [1] we know that output: 29*29
        # batch normalization
        self.bn2 = nn.BatchNorm2d(16)
        # ReLU activation after conv2
        self.relu2 = nn.ReLU() # output: 29*29
        # Max pooling layer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 14*14 (from the test in dim_images.ipynb)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1) # from [1] we know that output: 12*12
        # batch normalization
        self.bn3 = nn.BatchNorm2d(32)
        # ReLU activation after conv3  
        self.relu3 = nn.ReLU() # output: 12*12

        # Fully connected layer. 32: number of channels; 12, 12: height and width of the feature map
        self.fc = nn.Linear(32 * 7 * 7, 15)

        self.initialize_weights()


    def initialize_weights(self):
        for module in self.modules(): # self.modules() comes from nn.Module; to recursively iterate over all the modules
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01) # initial weights drawn from a Gaussian distribution having a mean of 0 and a standard deviation of 0.01
                init.constant_(module.bias, 0) # set the bias to 0


    # A model will have a forward() function
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(-1, 32 * 7 * 7)  # flatten the tensor before passing to fully connected layers (the size -1 is inferred from other dimensions)
        
        x = self.fc(x)

        return x





# CNN with batch normalization, different size of convolutional filters, dropout
class CNN3(nn.Module):

    # A model will have an __init__() function, where it instantiates its layers

    def __init__(self):
        super(CNN3, self).__init__() # the constructor of the parent class (nn.Module) is called to initialize the model properly.

        # Convolutional layer 1: in_channels=1 because we have a greyscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1) # from [1] we get the formula: output = ((input - kernel_size + 2*padding)/stride) + 1 => 62*62
        # batch normalization
        self.bn1 = nn.BatchNorm2d(8) # the parameter should be equal to the out_channels of the convolution
        # ReLU activation after conv1
        self.relu1 = nn.ReLU() # output: 62*62
        # Max pooling layer 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 31*31 (from 62/2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1) # from [1] we know that output: 29*29
        # batch normalization
        self.bn2 = nn.BatchNorm2d(16)
        # ReLU activation after conv2
        self.relu2 = nn.ReLU() # output: 29*29
        # Max pooling layer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 14*14 (from the test in dim_images.ipynb)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1) # from [1] we know that output: 12*12
        # batch normalization
        self.bn3 = nn.BatchNorm2d(32)
        # ReLU activation after conv3  
        self.relu3 = nn.ReLU() # output: 12*12
        # Dropout
        self.dp = nn.Dropout(.4) # dropout probability of 0.4
        # Fully connected layer. 32: number of channels; 12, 12: height and width of the feature map
        self.fc = nn.Linear(32 * 7 * 7, 15)

        self.initialize_weights()


    def initialize_weights(self):
        for module in self.modules(): # self.modules() comes from nn.Module; to recursively iterate over all the modules
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01) # initial weights drawn from a Gaussian distribution having a mean of 0 and a standard deviation of 0.01
                init.constant_(module.bias, 0) # set the bias to 0


    # A model will have a forward() function
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dp(x)

        x = x.view(-1, 32 * 7 * 7)  # flatten the tensor before passing to fully connected layers (the size -1 is inferred from other dimensions)
        
        x = self.fc(x)

        return x

