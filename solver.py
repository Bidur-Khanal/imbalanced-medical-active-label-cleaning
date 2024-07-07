import time
import torch
import numpy as np
from utils import AverageMeter, ProgressMeter, accuracy, Summary, loss_coteaching,  loss_coteaching_select_with_VOG_mixed
import pdb
from torch.autograd import Variable
import torch.nn as nn
from backpack import extend, backpack
from backpack.extensions import BatchGrad




def train(train_loader, model, criterion, optimizer, epoch, device, args, corrupt_indices, uncorrupt_indices):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))


    sample_loss_fn  = nn.CrossEntropyLoss(reduction = "none").to(device)

    all_losses = []
    targets = []

    # switch to train mode
    model.train()

    end = time.time()
    indexes = []
    for i, (images, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        targets.extend(target)

        if args.use_gpu:
            images = images.to(device)
            target = target.to(device)

        
        # compute output
        output = model(images)
        loss = criterion(output, target)
        sample_loss = sample_loss_fn(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    
        all_losses.extend(sample_loss.detach().cpu().numpy())
        indexes.extend(index)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg, np.array(indexes), np.array(all_losses), np.array(targets)




def inference(val_loader, model, criterion, device, args):
    all_targets = []
    all_predicted_targets = []
    all_predicted_probs = []
    def run_inference(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
           

            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.use_gpu:
                    images = images.to(device)
                    target = target.to(device)
            
                # compute output
                output = model(images)
                prob = torch.nn.functional.softmax(output, dim=1)
                all_predicted_probs.extend(prob.cpu().numpy())
                loss = criterion(output, target)
                _, predicted_target = output.max(1)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

                
                all_predicted_targets.extend(predicted_target.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_inference(val_loader)
    progress.display_summary()

    return top1.avg, top5.avg, losses.avg, all_targets, all_predicted_targets, np.array(all_predicted_probs)

def train_coteaching_general(train_loader, model1, model2, criterion, optimizer1, optimizer2, rate_schedule, epoch, device, args, corrupt_indices, uncorrupt_indices):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss Task1', ':.4e')
    losses2= AverageMeter('Loss Task2', ':.4e')

    top1_model1= AverageMeter('Acc@1 Task1', ':6.2f')
    top5_model1 = AverageMeter('Acc@5 Task1', ':6.2f')

    top1_model2= AverageMeter('Acc@1 Task2', ':6.2f')
    top5_model2 = AverageMeter('Acc@5 Task2', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses1, losses2, top1_model1, top5_model1, top1_model2, top5_model2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model1.train()
    model2.train()

    end = time.time()

    indexes = []
    all_guessed_uncorrupted = []
    all_losses1 = []
    all_saliencies = []
    targets = []
    

    for i, (images, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        targets.extend(target)

        if args.use_gpu:
            images = images.to(device)
            target = target.to(device)

        # compute output and accuracies for model 1
        output1 = model1(images)
        acc1_model1, acc5_model1 = accuracy(output1, target, topk=(1, 2))
        top1_model1.update(acc1_model1[0], images.size(0))
        top5_model1.update(acc5_model1[0], images.size(0))

         # compute output and accuracies for model 1
        output2 = model2(images)
        acc1_model2, acc5_model2 = accuracy(output2, target, topk=(1, 2))
        top1_model2.update(acc1_model2[0], images.size(0))
        top5_model2.update(acc5_model2[0], images.size(0))

        loss1, loss2, guessed_uncorrupted, sample_loss1 = loss_coteaching(output1, output2, target, criterion, rate_schedule[epoch], index, device, corrupt_indices, uncorrupt_indices)
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))

        # compute gradient and do SGD step, model 1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # compute gradient and do SGD step, model 2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        all_losses1.extend(sample_loss1.detach().cpu().numpy())
        indexes.extend(index)
        all_guessed_uncorrupted.extend(guessed_uncorrupted)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return top1_model1.avg, top5_model1.avg, losses1.avg, top1_model2.avg, top5_model2.avg, losses2.avg, np.array(all_saliencies), np.array(all_saliencies), np.array(indexes), all_guessed_uncorrupted, all_losses1, np.array(targets)

def inference_coteaching_general(val_loader, model1, model2, criterion, device, args):
    
    all_targets = []
    all_predicted_targets = []
    all_predicted_probs = []


    def run_inference(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
           

            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.use_gpu:
                    images = images.to(device)
                    target = target.to(device)
            
                # compute output model1 and model2
                output1 = model1(images)
                output2 = model2(images)

                outputs = output1+output2
                loss = torch.mean(criterion(outputs, target))
                acc1, acc5 = accuracy(outputs, target, topk=(1, 2))

                prob = torch.nn.functional.softmax(outputs, dim=1)
                all_predicted_probs.extend(prob.cpu().numpy())
                _, predicted_target = outputs.max(1)

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
                
                all_predicted_targets.extend(predicted_target.cpu().numpy())
                all_targets.extend(target.cpu().numpy())        

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ")

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    run_inference(val_loader)
    progress.display_summary()

    return top1.avg, top5.avg, losses.avg, all_targets, all_predicted_targets, np.array(all_predicted_probs)


def compute_gradient_for_data(train_loader, model, criterion, optimizer, device, args):

    '''
        Calculate gradients matrix on current network for specified training dataset.
    '''
    #criterion = extend(criterion)
    model.fc = extend(model.fc)

    # Initialize a matrix to save gradients. (on cpu)
    gradients = []
    indexes = []
    for i, (input, targets, index) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input.to(device))
        loss = criterion(outputs,targets.to(device))
        batch_num = targets.shape[0]

        with backpack(BatchGrad()):
            loss.backward()

        for name, param in model.named_parameters():
            if 'linear.weight' in name or 'fc.weight' in name:
                weight_parameters_grads = param.grad_batch
                
            elif 'linear.bias' in name or 'fc.bias' in name:
                bias_parameters_grads = param.grad_batch

        indexes.extend(index)

        if args.all_activations == 1:
            gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],dim=1).cpu().numpy())
        else:
            gradients.append(torch.cat([bias_parameters_grads[np.arange(input.shape[0]),targets].unsqueeze(1), 
                                        weight_parameters_grads[np.arange(input.shape[0]),targets].flatten(1)],dim=1).cpu().numpy())
        
    gradients = np.concatenate(gradients, axis=0)
    return gradients, indexes



def train_coteaching_simplified_mix_VOG(train_loader, model1, model2, criterion, optimizer1, optimizer2, rate_schedule, epoch, device, args, vog1, vog2, corrupt_indices, uncorrupt_indices):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses1 = AverageMeter('Loss Task1', ':.4e')
    losses2= AverageMeter('Loss Task2', ':.4e')

    top1_model1= AverageMeter('Acc@1 Task1', ':6.2f')
    top5_model1 = AverageMeter('Acc@5 Task1', ':6.2f')

    top1_model2= AverageMeter('Acc@1 Task2', ':6.2f')
    top5_model2 = AverageMeter('Acc@5 Task2', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses1, losses2, top1_model1, top5_model1, top1_model2, top5_model2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model1.train()
    model2.train()

    end = time.time()

    indexes = []
    all_guessed_uncorrupted = []
    all_losses1 = []
    targets = []
   

    for i, (images, target, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        targets.extend(target)
        if args.use_gpu:
            images = images.to(device)
            target = target.to(device)
       
        # compute output and accuracies for model 1
        output1 = model1(images)
        acc1_model1, acc5_model1 = accuracy(output1, target, topk=(1, 2))
        top1_model1.update(acc1_model1[0], images.size(0))
        top5_model1.update(acc5_model1[0], images.size(0))

         # compute output and accuracies for model 2
        output2 = model2(images)
        acc1_model2, acc5_model2 = accuracy(output2, target, topk=(1, 2))
        top1_model2.update(acc1_model2[0], images.size(0))
        top5_model2.update(acc5_model2[0], images.size(0))

        indexes.extend(index)

        loss1, loss2, guessed_uncorrupted, sample_loss1 = loss_coteaching_select_with_VOG_mixed(output1, output2, target, criterion, vog1, vog2, rate_schedule[epoch], index, device, args, epoch,corrupt_indices, uncorrupt_indices)
        losses1.update(loss1.item(), images.size(0))
        losses2.update(loss2.item(), images.size(0))

        # compute gradient and do SGD step, model 1
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        # compute gradient and do SGD step, model 2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # indices of examples guessed as the samples
        all_losses1.extend(sample_loss1.detach().cpu().numpy())
        all_guessed_uncorrupted.extend(guessed_uncorrupted)


        if i % args.print_freq == 0:
            progress.display(i + 1)

    return top1_model1.avg, top5_model1.avg, losses1.avg, top1_model2.avg, top5_model2.avg, losses2.avg, np.array(indexes),all_guessed_uncorrupted, all_losses1, np.array(targets)
