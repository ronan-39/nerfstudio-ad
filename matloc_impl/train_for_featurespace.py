import resnet_testing
from torchvision import datasets, models, transforms
import encoder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path
import sys
import time
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

model = resnet_testing.get_modified_model()
model.train()

transform = transforms.Compose([
    transforms.ToTensor(),
])


data_dir = Path.home().joinpath('Documents/nerfstudio-ad/matloc_impl/outputs/unnamed/feature_image_pairs')
# data_dir = Path.home().joinpath('Documents/nerfstudio-ad/matloc_impl/outputs/unicorn_torch/feature_image_pairs')
if not data_dir.exists():
    print("invalid data dir")
    sys.exit(1)

valid_size = 0.2
batch_size = 20
num_workers=8

training_set = encoder.EncodingDataset(data_dir)
validation_set = encoder.EncodingDataset(data_dir)

training_loader = DataLoader(training_set, batch_size=5, sampler=SubsetRandomSampler(list(range(training_set.len))), num_workers=0)
validation_loader = DataLoader(validation_set, batch_size=5, sampler=SubsetRandomSampler(list(range(validation_set.len))), num_workers=0)

print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/image_encoder_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 20

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1