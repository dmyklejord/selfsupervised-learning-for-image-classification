
import time
import numpy as np
import torch
import torch.nn as nn

from lightly.loss import NTXentLoss
from lightly.models.utils import update_momentum, batch_shuffle, batch_unshuffle

class NT_Xent(nn.Module):
    # from: https://github.com/Spijkervet/SimCLR/blob/cd85c4366d2e6ac1b0a16798b76ac0a2c8a94e58/simclr/modules/nt_xent.py

    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def train_moco(num_epochs, model, optimizer, device,
                train_loader, save_model,
                logging_interval=10,
                save_epoch_states=False):

    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    start_time = time.time()
    total_loss = 0

    criterion = NTXentLoss(memory_bank_size=4096)

    for epoch in range(num_epochs):
        print(f'Current model: {save_model}')
        model.train()

        for batch_idx, (x_query, x_key, _) in enumerate(train_loader):
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)

            x_query = x_query.to(device)
            x_key = x_key.to(device)
            query = model(x_query)
            key = model.forward_momentum(x_key)

            loss = criterion(query, key)
            total_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        log_dict['train_loss_per_epoch'].append(total_loss.item())

        if save_epoch_states:
            torch.save(model.state_dict(), save_model+'-{}.pt'.format(epoch))


    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model+'.pt')

    return log_dict

def train_simclr(num_epochs, model, optimizer, device,
                train_loader, save_model,
                logging_interval=10,
                save_epoch_states=False):

    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    start_time = time.time()
    total_loss = 0

    # criterion = NTXentLoss()
    criterion = NT_Xent(train_loader.batch_size, temperature=0.5, world_size=1)

    for epoch in range(num_epochs):
        print(f'Current model: {save_model}')

        for batch_idx, (x_i, x_j, _) in enumerate(train_loader):

            x_i = x_i.to(device)
            x_j = x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            total_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        log_dict['train_loss_per_epoch'].append(total_loss.item())

        if save_epoch_states:
            torch.save(model.state_dict(), save_model+'-{}.pt'.format(epoch))


    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model+'.pt')

    return log_dict

def train_byol(num_epochs, model, optimizer, device,
                train_loader, save_model,
                logging_interval=100,
                save_epoch_states=False):

    from lightly.loss import NegativeCosineSimilarity
    from lightly.models.utils import update_momentum

    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    start_time = time.time()
    total_loss = 0

    criterion = NegativeCosineSimilarity()
    for epoch in range(num_epochs):
        print(f'Current model: {save_model}')

        for batch_idx, (x0, x1, _) in enumerate(train_loader):

            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x0 = x0.to(device)
            x1 = x1.to(device)

            # Goes through twice for the symmetrical loss:
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)

            # The symmetrical loss itself:
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        log_dict['train_loss_per_epoch'].append(total_loss.item())

        if save_epoch_states:
            torch.save(model.state_dict(), save_model+'-{}.pt'.format(epoch))


    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model+'.pt')

    return log_dict
