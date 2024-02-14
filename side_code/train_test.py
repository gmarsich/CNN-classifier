import torch
from torch.utils.data import Dataset

from torchvision import transforms

from sklearn.metrics import confusion_matrix
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd





# To transform the images in the proper format
def transform(image):
    resize = transforms.Compose([transforms.Resize([64,64]), # resize the images as 64x64
                                 transforms.ToTensor(), # to get a PyTorch tensor. The pixels' values will be scaled from the range [0, 255] to the range [0.0, 1.0]
                                 transforms.Grayscale()]) # the image will just have one channel
    resized_image = resize(image)
    resized_image = resized_image * 255.0 # revert the normalization, i.e., go back to range [0, 255]
    return resized_image




# To transform the images in the proper format
def transform_ALEXNET(image):
    resize = transforms.Compose([transforms.Resize([224,224]), # resize the images as indicated by the original paper
                                 transforms.ToTensor()]) # to get a PyTorch tensor. The pixels' values will be scaled from the range [0, 255] to the range [0.0, 1.0]
    resized_image = resize(image)
    #resized_image = resized_image * 255. # if uncommented, the accuracy will decrease
    return resized_image




# To horizontally flip the images
class HorizontallyFlippedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.horizontal_flip_transform = transforms.RandomHorizontalFlip(p=1)

    def __getitem__(self, index):
        image, label = self.original_dataset[index]
        flipped_image = self.horizontal_flip_transform(image)
        return flipped_image, label

    def __len__(self):
        return len(self.original_dataset)




# To train for just one epoch
def train_one_epoch(model, loader, optimizer, loss_function): # to train for just one epoch (one epoch: the network sees the whole training set)
    running_loss = 0.

    for i, data in enumerate(loader):

        inputs, labels = data # get the minibatch

        outputs = model(inputs) # forward pass

        loss = loss_function(outputs, labels) # compute the loss
        running_loss += loss.item() # sum up the loss for the minibatches processed so far

        optimizer.zero_grad() # notice that by default, the gradients are accumulated, hence we need to set them to zero
        loss.backward() # backward pass
        optimizer.step() # update the weights

    return running_loss/(i+1) # average loss per minibatch




# To perform the training
def train_model(model, EPOCHS, train_loader, val_loader, optimizer, loss_function):

    best_validation_loss = np.inf

    loss_train = []  # store the values of the loss for the training set
    loss_val = []    # store the values of the loss for the validation set
    accuracies_train = []  # store the values of the accuracy for the training set
    accuracies_val = []    # store the values of the accuracy for the validation set

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function)
        loss_train.append(train_loss)

        running_validation_loss = 0.0

        # If using dropout and/or batch normalization, set the model to evaluation mode for validation
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():  # disable gradient computation and reduce memory consumption
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_validation_loss += vloss
                _, predicted = torch.max(voutputs.data, 1)
                total += vlabels.size(0)
                correct += (predicted == vlabels).sum().item()

            accuracy = 100 * correct / total
            accuracies_val.append(accuracy)

        validation_loss = running_validation_loss / (i + 1)  # average validation loss per minibatch
        loss_val.append(validation_loss)

        print('LOSS: train: {}, validation: {}; accuracy validation set: {}%\n'.format(train_loss, validation_loss,
                                                                                        accuracy))

        # Track best performance (based on validation), and save the model
        if validation_loss < best_validation_loss:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_validation_loss = validation_loss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        # Evaluate on the training set to calculate accuracy
        model.eval()

        running_train_correct = 0
        running_train_total = 0

        with torch.no_grad():
            for i, tdata in enumerate(train_loader):
                tinputs, tlabels = tdata
                toutputs = model(tinputs)
                _, predicted = torch.max(toutputs.data, 1)
                running_train_total += tlabels.size(0)
                running_train_correct += (predicted == tlabels).sum().item()

            train_accuracy = 100 * running_train_correct / running_train_total
            accuracies_train.append(train_accuracy)

    return model_path, loss_train, loss_val, accuracies_train, accuracies_val




# To compute the accuracy on the test dataset and build the confusion matrix
def perform_test(model, model_path, test_loader, dataset_test):
    correct_test = 0
    total_test = 0

    y_pred = []
    y_true = []

    model.load_state_dict(torch.load(model_path)) # loads the saved weights from the specified path
    model.eval()

    with torch.no_grad():

        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            y_pred.extend(predicted)
            y_true.extend(labels) 


    accuracy_test = 100 * correct_test / total_test
    print('Accuracy of the network on the test images: %d %%' % (accuracy_test))


    # Build confusion matrix
    classes = dataset_test.classes
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (5,5))
    sn.heatmap(df_cm, annot=False)




# To extract features for each sample in the dataset
def extract(model, loader):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            # Extract features from the intermediate layer
            interm_feat = model(inputs)
            features.append(interm_feat.view(interm_feat.size(0), -1).numpy())
            labels.append(target.numpy())

    # Concatenate features and labels
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels