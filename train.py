from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

class TrainHelper:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train_model(self, model, device, train_loader, optimizer, loss_fn='nll_loss'):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            
            # Move to device
            data, target = data.to(device), target.to(device)
            # print(type(data))

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            
            
            # Predict
            y_pred = model(data)

            # Calculate loss
            if loss_fn == 'cross_entropy':
                loss = F.cross_entropy(y_pred, target)
            elif loss_fn == 'nll_loss':
                loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

    def test_model(self, model, device, test_loader, loss_fn='nll_loss'):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # Move to device
                data, target = data.to(device), target.to(device)
                output = model(data)
                if loss_fn == 'cross_entropy':
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                elif loss_fn == 'nll_loss':
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        
    def plot(self):
        t = [t_items.item() for t_items in self.train_losses]
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(t)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
    
    def train_for_epoch(self, model, device, train_loader, test_loader, optimizer, scheduler=None, epoch=50, loss_fn='nll_loss'):
        for i in range(epoch):
            if(scheduler):
                current_lr = scheduler.get_lr()
                print("Current LR from Scheduler: "+str(current_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            print("EPOCH:", i+1)
            self.train_model(model, device, train_loader, optimizer, loss_fn)
            self.test_model(model, device, test_loader, loss_fn)
            if(scheduler):
                current_lr = scheduler.step()
                
            
    
