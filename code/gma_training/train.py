import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pdb
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--log', type=str, default='convergence.csv', help='name of log file')
parser.add_argument('--data', type=str, default='', choices=[
    'msk_lung_egfr',
    'msk_lung_tp53',
    'msk_lung_kras',
    'msk_lung_stk11',
    'msk_lung_alk',
    'msk_lung_io',
    'sinai_breast_cancer',
    'sinai_breast_er',
    'sinai_breast_pr',
    'sinai_breast_her2',
    'sinai_lung_egfr',
    'sinai_ibd_detection',
    'sinai_bladder_cancer',
    'sinai_colorectal_cancer',
    'sinai_dcis_cancer',
    'sinai_kidney_cancer',
    'sinai_oral_cancer',
    'sinai_prostate_cancer',
    'sinai_thyroid_cancer',
    'biome_breast_hrd'
], help='which data to use')
parser.add_argument('--encoder', type=str, default='', choices=[
    'tres50_imagenet',
    'ctranspath',
    'phikon',
    'uni',
    'virchow',
    'gigapath',
    'dinosmall',
    'dinobase'
], help='which encoder to use')
parser.add_argument('--mccv', default=1, type=int, choices=list(range(1,21)), help='which seed (default: 1/20)')
#OPTIMIZATION PARAMS
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')

def main():
    
    # Get user input
    global args
    args = parser.parse_args()

    # Set datasets
    train_dset, val_dset = datasets.get_datasets(mccv=args.mccv, data=args.data, encoder=args.encoder)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Dim of features
    if args.encoder.startswith('tres50'):
        args.ndim = 1024
    elif args.encoder == 'ctranspath':
        args.ndim = 768
    elif args.encoder == 'phikon':
        args.ndim = 768
    elif args.encoder == 'uni':
        args.ndim = 1024
    elif args.encoder == 'virchow':
        args.ndim = 2560
    elif args.encoder == 'gigapath':
        args.ndim = 1536
    elif args.encoder.startswith('dinosmall'):
        args.ndim = 384
    elif args.encoder.startswith('dinobase'):
        args.ndim = 768
    
    # Get model
    gma = modules.GMA(dropout=True, n_classes=2, ndim=args.ndim)
    gma.cuda()
    
    # Set loss
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Set optimizer
    params_groups = get_params_groups(gma)
    optimizer = optim.AdamW(params_groups)
    
    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True
        
    # Set output files
    with open(os.path.join(args.output,args.log), 'w') as fconv:
        fconv.write('epoch,metric,value\n')
    
    # Main training loop
    for epoch in range(args.nepochs+1):
        
        ## Training logic
        if epoch > 0:
            loss = train(epoch, train_loader, gma, criterion, optimizer, lr_schedule, wd_schedule)
            print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch, args.nepochs, loss))
            with open(os.path.join(args.output,args.log), 'a') as fconv:
                fconv.write('{},loss,{}\n'.format(epoch, loss))
        
        ## Validation logic
        probs = test(epoch, val_loader, gma)
        auc = roc_auc_score(val_loader.dataset.df.target, probs)
        ### Printing stats
        print('Validation\tEpoch: [{}/{}]\tAUC: {}'.format(epoch, args.nepochs, auc))
        with open(os.path.join(args.output,args.log), 'a') as fconv:
            fconv.write('{},auc,{}\n'.format(epoch, auc))
        
        ### Model saving logic
        obj = {
            'epoch': epoch,
            'state_dict': gma.state_dict(),
            'auc': auc,
            'optimizer' : optimizer.state_dict()
        }
        torch.save(obj, os.path.join(args.output,'checkpoint.pth'))

def test(run, loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader)).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run, args.nepochs, i+1, len(loader)))
            ## Copy batch to GPU
            input = input.squeeze(0).cuda()
            ## Forward pass
            _, _, output = model(input)
            output = F.softmax(output, dim=1)
            ## Clone output to output vector
            probs[i] = output.detach()[:,1].item()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, (input, target) in enumerate(loader):
        ## Update weight decay and learning rate according to their schedule
        it = len(loader) * (run-1) + i # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        ## Copy to GPU
        input = input.squeeze(0).cuda()
        target = target.long().cuda()
        ## Forward pass
        _, _, output = model(input)
        ## Calculate loss
        loss = criterion(output, target)
        ## Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Store loss
        running_loss += loss.item()
        print('Training\tEpoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {}'.format(run, args.nepochs, i+1, len(loader), loss.item()))
    return running_loss / len(loader)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    main()
