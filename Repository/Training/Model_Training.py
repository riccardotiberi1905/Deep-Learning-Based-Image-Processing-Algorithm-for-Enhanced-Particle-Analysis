from MSR_NET import MSRNet, count_parameters, MSRNet_2_classes, MSRNet_TL
from Datasets_Training import Dataset_Leyre, Dataset_Ricc_1_class, Dataset_Ricc_2_classes, Dataset_common
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from types import SimpleNamespace
from tloss import TLoss

# ---------------------- Loss Definitions ----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', task_type='multi-class', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        if task_type == 'multi-class' and alpha is not None:
            # alpha can be a list or tensor of class weights
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W), targets: (N, H, W)
        N, C, H, W = inputs.shape
        # Flatten predictions and targets
        logits = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.view(-1)

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        logp = torch.log(probs + 1e-12)

        # One-hot encode targets
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets_flat.unsqueeze(1), 1)

        # Focal weight term
        p_t = (probs * one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy
        ce_loss = - (one_hot * logp).sum(dim=1)

        # Class weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets_flat]
            ce_loss = alpha_t * ce_loss

        # Apply focal weighting
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, C, H, W), targets: (N, H, W) with class indices
        num_classes = logits.shape[1]
        # one-hot encode targets
        one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
        dims = (0, 2, 3)

        intersection = torch.sum(probs * one_hot, dims)
        cardinality  = torch.sum(probs + one_hot, dims)
        dice_score   = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_ce + self.alpha * loss_dice

if __name__ == '__main__':

    '''
    if __name__ == '__main__': Guard:
    This condition ensures that the code inside it runs only when the script 
    is executed directly, not when it’s imported by another process. 
    This prevents unwanted re-execution of the main training code in the child processes.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #========== Define parameters here =============================
    # log file
    if not os.path.exists('./logs_ricc_focal_loss_v2'):
        os.mkdir('./logs_ricc_focal_loss_v2')

    if not os.path.exists('./checkpoint_ricc_focal_loss_v2'):
        os.mkdir('./checkpoint_ricc_focal_loss_v2')


    # Hyperparameters
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    total_epoch = 40
    val_portion = 0.1
    learning_rate = 0.001
    batch_size = 128
    weight_decay = 1e-4
    check_path = 'net.pt'
    resume = False
    alpha = 0.5

    # Lists to record losses for plotting later
    Training_loss = []
    Validation_loss = []

    # ---------------------- Model Setup ------------------------
    net = MSRNet_2_classes()
    print("Total number of parameters: "+str(count_parameters(net)))
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    #for name, param in net.named_parameters():
    #    print(name, param.device)

    # --------------------- Loss, Optimizer, Scheduler -----------------
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = CombinedLoss(alpha=alpha)
    class_weights = [1.0, 2.0, 2.0]
    criterion = FocalLoss(gamma=2.0,
                          alpha=class_weights,
                          reduction='mean',
                          task_type="multi-class",
                          num_classes=3)
    #IMAGE_SIZE = 64
    #tloss_config = SimpleNamespace(
    #data = SimpleNamespace(image_size=IMAGE_SIZE),
    #device = device
    #)
    #criterion = TLoss(tloss_config, nu=1.0, epsilon=1e-8, reduction="mean")
    optimizer = optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adam([
    #{"params": net.parameters(), "lr": learning_rate, "name": "model"},
    #{"params": criterion.parameters(), "lr": learning_rate, "name": "tloss"}], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5) #minimize loss

    # ---------------------- Data Setup -----------------------
    train_dataset = Dataset_Ricc_2_classes()
    print("Total Data:", train_dataset.__len__())

    # Split into train / validation partitions
    n_val = int(len(train_dataset) * val_portion)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=16
    )

    # --------------------- Checkpoint resume -------------------
    best_loss = np.Inf

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint_ricc_focal_loss_v2..')
        assert os.path.isdir('checkpoint_ricc_focal_loss_v2'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint_ricc_focal_loss_v2/'+check_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']


    # ---------------------- Training Function ----------------------
    def train(epoch):

        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        # train network
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # If using DataParallel, access the original module via net.module
            #conv1_weight = net.module.down_conv1[0].weight if isinstance(net, torch.nn.DataParallel) else net.down_conv1[
            #    0].weight
            #print("down_conv1 weight device:", conv1_weight.device)
            #break
            inputs, targets = inputs.to(device), targets.to(device)
            #print("Inputs device:", inputs.device)
            #print("Input shape:", inputs.shape)
            targets = torch.squeeze(targets)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #probs = torch.sigmoid(outputs)
            #loss = criterion(probs, targets)
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print("Epoch %d: Train loss %.4f" % (epoch, avg_train_loss))
        Training_loss.append(avg_train_loss)

        #  Save checkpoint for training progress in logs directory
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),  # Save optimizer state
            'scheduler': scheduler.state_dict(),  # Save scheduler state (if needed)
            'best_loss': avg_train_loss,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],  # Current learning rate
            'Training_loss': Training_loss,
            'Validation_loss': Validation_loss,
            # Optionally, add any other metrics you are tracking
        }

        torch.save(state, os.path.join('./logs_ricc_focal_loss_v2', f'net_{epoch}.pt'))


    # ---------------------- Evaluation Function ----------------------
    def test(epoch, display=False):
        global best_loss
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = torch.squeeze(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                #loss = criterion(torch.sigmoid(outputs), targets)

                test_loss += loss.item()

        print('Epoch %d: Valid loss: %.4f' % (epoch, test_loss))
        Validation_loss.append(test_loss)
        avg_val_loss = test_loss / len(val_loader)
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save checkpoint.
        if avg_val_loss < best_loss:
            print('Saving best model checkpoint..')
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),  # Save optimizer state
                'scheduler': scheduler.state_dict(),  # Save scheduler state (if needed)
                'best_loss': avg_val_loss,
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],  # Current learning rate
                'Training_loss': Training_loss,
                'Validation_loss': Validation_loss,
                # Optionally, add any other metrics you are tracking
            }
            torch.save(state, './checkpoint_ricc_focal_loss_v2/' + check_path)
            best_loss = avg_val_loss

    # --------------- Main Training Loop ---------------------
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        test(epoch, False)

    # ---------------------- Plotting Results ----------------------
    epochs = range(start_epoch, total_epoch)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, Training_loss, label='Training Loss')
    plt.plot(epochs, [v / len(val_loader) for v in Validation_loss], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()