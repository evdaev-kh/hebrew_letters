import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
import numpy as np
from sklearn import metrics

#Define the class to Hebrew letter map
class_to_letter = {
    0: 'א',
    1: 'ב',
    2: 'ג',
    3: 'ד',
    4: 'ה',
    5: 'ו',
    6: 'ז',
    7: 'ח',
    8: 'ט',
    9: 'י',
    10: 'כ',
    11: 'ך',
    12: 'ל',
    13: 'מ',
    14: 'ם',
    15: 'נ',
    16: 'ן',
    17: 'ס',
    18: 'ע',
    19: 'פ',
    20: 'ף',
    21: 'צ',
    22: 'ץ',
    23: 'ק',
    24: 'ר',
    25: 'ש',
    26: 'ת' 
}

# Define the CNN model
class MultiClassCNNModel(nn.Module):
    def __init__(self, in_channels, num_classes, debug=False, dim_size=(32, 32)):
        """ 
            CNN model using 2 Conv2D blocks with 2 max pool layers for multi-class classification

            Params:
                in_channels: # of channels for the image (usually 3)
                num_classes: # number of classes to predict
                debug:      a flag to debug some info about the outputs of each layer
                dim_size:   the x and y dimension of the image (img size = X x Y x # channels)
        """
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)

        # # Convolution 3
        # self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.relu3 = nn.ReLU()
        # Max pool 2
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)
        
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(3136, 64)

        # self.fc1 = nn.Linear(89888, num_classes)
        self.fc2 = nn.Linear(64, num_classes)

        self.debug = debug

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)

        if self.debug:
            print(f"cnn1: {out.shape}")

        out = self.relu1(out)
        if self.debug:
            print(f"relu1: {out.shape}")
        # Max pool 1
        out = self.maxpool1(out)
        if self.debug:
            print(f"mp1: {out.shape}")
        
        # Convolution 2
        out = self.cnn2(out)
        if self.debug:
            print(f"cnn2: {out.shape}")
        out = self.relu2(out)
        if self.debug:
            print(f"relu2: {out.shape}")
        # Max pool 2
        out = self.maxpool2(out)
        if self.debug:
            print(f"mp2: {out.shape}")
        
        # Convolution 3
        out = self.cnn3(out)
        if self.debug:
            print(f"cnn3: {out.shape}")
        out = self.relu3(out)
        if self.debug:
            print(f"relu3: {out.shape}")
        # Max pool 3
        out = self.maxpool3(out)
        if self.debug:
            print(f"mp3: {out.shape}")
        
        # Resize
        out = out.view(out.size(0), -1)
        if self.debug:
            print(f"resize: {out.shape}")
        # Linear function (readout)
        out = self.fc1(out)
        if self.debug:
            print(f"linear: {out.shape}")
            print(out.shape)
        
        out = self.fc2(out)
        # softmax_output = nn.Softmax(dim=1)(out)
        # _, predicted = torch.max(softmax_output, 1)
        return out
    

if __name__ == "__main__":

    # Path to the training dataset
    training_dir = "./hhd_dataset/TRAIN"

    # Path to the testing dataset
    testing_dir = "./hhd_dataset/TEST"

    # Define the transformations for the images
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #Load training and testing datasets
    train_dataset = datasets.ImageFolder(root=training_dir, transform=transform)
    testing_dataset = datasets.ImageFolder(root=testing_dir, transform=transform)


    #Data loader for training loop
    train_loader = DataLoader( train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader( testing_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    #Define the CNN model
    cnn_model = MultiClassCNNModel(num_classes=len(train_dataset.classes))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    debug = False
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            if debug:
                print(f"Images = {len(images[0])}")
                print(f"Label = {len(labels)}")
                print(images)
                print(labels)
            # Forward pass
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            
            if debug:
                print(outputs)
                print(labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    
    #Test the accuracy on the testing dataset
    total = 0
    correct = 0
    scores = {}
    debug = False
    predicted_ = []
    y_true = []
    with torch.no_grad():    
        for i, (images, labels) in enumerate(test_loader):
                
                if debug:
                    print(f"Images = {len(images[0])}")
                    print(f"Label = {len(labels)}")
                    print(images)
                    print(labels)
                # Forward pass
                predicted = cnn_model(images)
                # loss = criterion(predicted, labels)

                softmax_output = nn.Softmax(dim=1)(predicted)
                _, predicted = torch.max(softmax_output, 1)

                # if i == 0:
                #      predicted_ = predicted
                #      print(predicted_)
                # else:
                #      predicted_ = torch.cat([predicted_], predicted)

                predicted_.append(predicted)
                y_true.append(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(testing_dataset)} test images: %d %%' % (100 * correct / total))
    
    y_pred = []
    for t in predicted_:
        for e in t:
            y_pred.append(e)
        
    Y_true = []

    for t in y_true:
        for e in t:
            Y_true.append(e)
    
    letters = []
    for v in class_to_letter.values():
        # print(v)
        letters.append(v)
    
    # Create the confusion matrix
    conf_matrix = metrics.confusion_matrix(Y_true, y_pred, normalize='true')
    fig = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=letters).plot()
    plt.title(f"Confusion Matrix for SVM with Accuracy Score {metrics.accuracy_score(Y_true, y_pred)}")
    plt.savefig("my_plot_transparent.png", transparent=True)  # Save with transparent background
    plt.tight_layout()
    plt.show()