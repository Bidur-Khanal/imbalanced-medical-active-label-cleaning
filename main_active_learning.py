import os
import os.path
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import argparse
import utils 
from solver import  inference, train
import neptune as neptune
import matplotlib.pyplot as plt
from neptune.types import File
from utils import custom_ISIC_faster, custom_hyper_kvasir_faster, custom_DRD_faster, custom_Imbalanced_Histopathology_faster
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef as mcc
import torch.utils.data as data
from active_learning_strategies import EntropySampling, LeastConfidence, MarginSampling, BALDDropout, Coreset
import torch.backends.cudnn as cudnn
from LNL import CoteachingLNL, CoteachingVOGLNL, CrossEntropy


## Set Seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    cudnn.deterministic = True


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

################################################################
def main(args):

    # Preprocessings the data
    print('==> Preparing data..')

    # Augmentations used 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5),sigma=(0.1, 1.0))

    # Define the Color Jitter transform
    color_jitter = transforms.ColorJitter(
        brightness=0.1,  # Minimally change brightness by 10%
        contrast=0.1,    # Minimally change contrast by 10%
        saturation=0.1,  # Minimally change saturation by 10%
        hue=0.01         # Minimally change hue by 1%
    )

    train_transform =  transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply([gaussian_blur], p=0.5),
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])


    val_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
            ])


    if args.dataset == "isic":

        NUM_CLASSES = 8
        train_dataset = custom_ISIC_faster(split_type="train", transform = train_transform)
        val_dataset = custom_ISIC_faster(split_type="val", transform = val_transform)
        test_dataset = custom_ISIC_faster(split_type="test", transform = val_transform)

        correct_labels = train_dataset.targets.copy()
        corrupt_indices, uncorrupt_indices = corrupt_labels(train_dataset.targets, NUM_CLASSES, args.corrupt_prob, uncorrupt_classes, args)

        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        valloader = FastDataLoader(val_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)


    if args.dataset == "drd":

        NUM_CLASSES = 5
        train_dataset = custom_DRD_faster(split_type="train", transform = train_transform)
        val_dataset = custom_DRD_faster(split_type="val", transform = val_transform)
        test_dataset = custom_DRD_faster(split_type="test", transform = val_transform)

        correct_labels = train_dataset.targets.copy()
        corrupt_indices, uncorrupt_indices = corrupt_labels(train_dataset.targets, NUM_CLASSES, args.corrupt_prob, uncorrupt_classes, args)

        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        valloader = FastDataLoader(val_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)


    if args.dataset == "kvasir":

        NUM_CLASSES = 7

        train_dataset = custom_hyper_kvasir_faster(split_type="train", transform = train_transform)
        val_dataset = custom_hyper_kvasir_faster(split_type="val", transform = val_transform)
        test_dataset = custom_hyper_kvasir_faster(split_type="test", transform = val_transform)

        correct_labels = train_dataset.targets.copy()
        corrupt_indices, uncorrupt_indices = corrupt_labels(train_dataset.targets, NUM_CLASSES, args.corrupt_prob, uncorrupt_classes, args)

        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        valloader = FastDataLoader(val_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)

    if args.dataset == "imbalanced_histopathology":


        NUM_CLASSES = 9
       
        train_dataset = custom_Imbalanced_Histopathology_faster(split_type="train", transform = train_transform)
        val_dataset = custom_Imbalanced_Histopathology_faster(split_type="val", transform = val_transform)
        test_dataset = custom_Imbalanced_Histopathology_faster(split_type="test", transform = val_transform)

        correct_labels = train_dataset.targets.copy()
        corrupt_indices, uncorrupt_indices = corrupt_labels(train_dataset.targets, NUM_CLASSES, args.corrupt_prob, uncorrupt_classes, args)

        # encapsulate data into dataloader form
        trainloader = FastDataLoader(train_dataset,batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)
        valloader = FastDataLoader(val_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
        testloader = FastDataLoader(test_dataset,batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)



    ## LNL epoch is greater than 0, we have to either train model using LNL or use existing LNL trained model
    if args.LNL_epochs > 0 :

        # we use dual model for LNL approach
        if args.arch == "resnet18":
            if args.pretrained_model == "pretrained":
                model1 = models.resnet18(pretrained=True)
                num_features1 = model1.fc.in_features
                new_linear_layer1 = nn.Linear(num_features1, NUM_CLASSES)
                model1.fc = new_linear_layer1

                model2 = models.resnet18(pretrained=True)
                num_features2 = model2.fc.in_features
                new_linear_layer2 = nn.Linear(num_features2, NUM_CLASSES)
                model2.fc = new_linear_layer2
            else:
                model1 = models.resnet18(pretrained=False, num_classes = NUM_CLASSES)
                model2 = models.resnet18(pretrained=False, num_classes = NUM_CLASSES)

        if args.use_gpu:
            model1 = model1.to(device)
            model2 = model2.to(device)


        # robustly train with LNL method 
        if args.method == "coteaching":
            ## check if there is already a saved Coteaching model and the guessed indices
            if args.pretrained_model is not None:
                Experiment_Name = args.dataset+"/coteaching/"+args.arch+"/"+args.pretrained_model+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)
            else:
                Experiment_Name = args.dataset+"/coteaching/"+args.arch+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)

            path = os.path.join(args.save_dir, args.dataset, Experiment_Name,'checkpoint_epoch_'+str(args.LNL_epochs)+'.pth.tar')

            if os.path.exists(path):

                if not args.scratch_after_LNL:
                    ## load the model
                    checkpoint = torch.load(path, map_location=device)
                    model1.load_state_dict(checkpoint['state_dict'])

                ## load the saved indices
                indices_path = os.path.join(args.save_dir, args.dataset, Experiment_Name,'LNL_correct_guessed_label_indices_epochs_'+str(args.LNL_epochs)+'.npy')
                guessed_clean_indices = np.load(indices_path, allow_pickle = True)
            else:
                # train LNL model
                LNL_coteaching = CoteachingLNL(model1, model2, device, args)
                guessed_clean_indices = LNL_coteaching.train(trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices,\
                                    train_dataset.targets, NUM_CLASSES, run) 


        elif args.method == "coteaching_VOG":
            ## check if there is already a saved Coteaching model and the guessed indices
            if args.pretrained_model is not None:
                Experiment_Name = args.dataset+"/VOG/"+args.arch+"/"+args.pretrained_model+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)+"_mix_ratio_"+str(args.mix_ratio)
            else:
                Experiment_Name = args.dataset+"/VOG/"+args.arch+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)+"_mix_ratio_"+str(args.mix_ratio)

            path = os.path.join(args.save_dir, args.dataset, Experiment_Name,'checkpoint_epoch_'+str(args.LNL_epochs)+'.pth.tar')

            if os.path.exists(path):
                if not args.scratch_after_LNL:
                    ## load the model
                    checkpoint = torch.load(path, map_location=device)
                    model1.load_state_dict(checkpoint['state_dict'])
                ## load the saved indices
                indices_path = os.path.join(args.save_dir, args.dataset, Experiment_Name,'LNL_correct_guessed_label_indices_epochs_'+str(args.LNL_epochs)+'.npy')
                guessed_clean_indices = np.load(indices_path, allow_pickle = True)
            else:
                LNL_coteaching_VOG = CoteachingVOGLNL(model1, model2, device, args)
                guessed_clean_indices = LNL_coteaching_VOG.train(trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices,\
                                    train_dataset.targets, NUM_CLASSES, run) 

        else:
            raise AssertionError('Specify the available LNL method')

        
    # non-LNL method 
    else:

        # if we don't use LNL approach, use just a single model
        if args.pretrained_model == "pretrained":
            model1 = models.resnet18(pretrained=True)
            num_features = model1.fc.in_features
            new_linear_layer = nn.Linear(num_features, NUM_CLASSES)
            model1.fc = new_linear_layer
        else:
            model1 = models.resnet18(pretrained=False, num_classes = NUM_CLASSES)
        if args.use_gpu:
            model1 = model1.to(device)


        if args.method == "cross_entropy":
            # LNL epoch is 0, then we have to train the model from scratch using the cross-entropy or load the saved model trained with cross-entropy
            if args.pretrained_model is not None:
                Experiment_Name = args.dataset+"/"+args.arch+"/"+args.pretrained_model+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)
            else:
                Experiment_Name = args.dataset+"/"+args.arch+"/"+"corrupt_prob_"+str(args.corrupt_prob)+"_seed_"+str(args.seed)

            path = os.path.join(args.save_dir, args.dataset, Experiment_Name,'checkpoint_best.pth.tar')

            if os.path.exists(path):
                ## load the model
                checkpoint = torch.load(path, map_location=device)
                model1.load_state_dict(checkpoint['state_dict'])
                
            else:
                # train LNL model
                CE_method = CrossEntropy(model1,device,args)
                CE_method.train(trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices,\
                                    train_dataset.targets, NUM_CLASSES, run) 
                
        elif args.method == "scratch":
            pass
        else:
            raise AssertionError('Specify the available method')


        
    ######### start the active labeling / active label cleaning round ##########
    currrent_budget = 0
    
    for round in range(args.label_cleaning_rounds):

        # stores values of last few epochs of the test set
        last_accuracies = []
        last_precisions = []
        last_recalls = []
        last_AUROC = []
        last_F1scores = []
        last_balanced_accuracies = []
        last_mcc = []

        # best values of validation set at start 
        best_acc1 = 0.
        best_precision_val = 0.
        best_recall_val = 0.
        best_AUROC_val = 0.
        best_F1score_val = 0.
        best_balanced_accuracy_val = 0.
        best_mcc_val = 0.
        
        # define scheduler and optimizers
        avg_criterion  = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model1.parameters()), args.lr_AL,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.label_cleaning_rounds)
    
        
        # active labeling / active label cleaning parameters
        all_indices = set(np.arange(len(train_dataset)))

        # at round 1, we decide if we want to use the clean samples guessed by LNL OR 
        # at round 0, if we don't want to train on our own not using LNL model
        if (round == 1 and not args.scratch_after_LNL) or (args.scratch_after_LNL and round == 0):
            print ("scratch after LNL")
            if args.use_LNL_cleaned_labels:
                cleaned_sample_indices = [x.item() for x in guessed_clean_indices]
            else:
                cleaned_sample_indices = []


        # from round 1, we start to clean other labels
        if round >= 1 or (args.scratch_after_LNL and round == 0):

            # separate noisy labels from guessed clean labels (if present)
            noisy_label_indices = list(all_indices - set(cleaned_sample_indices))

            # don't rely on any noisy labels begin relabeling from scratch
            if (args.active_learning_only) and (round == 1):
                my_random_state = np.random.RandomState(args.seed) # always use the same random subset
                new_clean_sample_indices = my_random_state.choice(np.array(noisy_label_indices), size=args.round_budget, replace=False).tolist()
                cleaned_sample_indices.extend(new_clean_sample_indices)

            else:

                if  (args.scratch_after_LNL and round == 0):
                    print ("scratch after LNL round 0")
                    pass               
               
                elif args.active_selection_method == "random":
                    my_random_state = np.random.RandomState(args.seed) # always use the same random subset
                    new_clean_sample_indices = my_random_state.choice(np.array(noisy_label_indices), size=args.round_budget, replace=False).tolist()
                    cleaned_sample_indices.extend(new_clean_sample_indices)
                

                elif args.active_selection_method == "entropy_sampling":
                    
                    noisy_trainloader = FastDataLoader(train_dataset, sampler = noisy_label_indices, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
                    new_clean_sample_indices = EntropySampling(noisy_trainloader,model1,noisy_label_indices,args.round_budget,device,args)
                    cleaned_sample_indices.extend(new_clean_sample_indices)

                elif args.active_selection_method == "least_confidence_sampling":
                    
                    noisy_trainloader = FastDataLoader(train_dataset, sampler = noisy_label_indices, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
                    new_clean_sample_indices = LeastConfidence(noisy_trainloader,model1,noisy_label_indices,args.round_budget,device,args)
                    cleaned_sample_indices.extend(new_clean_sample_indices)

                elif args.active_selection_method == "margin_sampling":
                    
                    noisy_trainloader = FastDataLoader(train_dataset, sampler = noisy_label_indices, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
                    new_clean_sample_indices = MarginSampling(noisy_trainloader,model1,noisy_label_indices,args.round_budget,device,args)
                    cleaned_sample_indices.extend(new_clean_sample_indices)

                elif args.active_selection_method == "BALD_sampling":
                    
                    noisy_trainloader = FastDataLoader(train_dataset, sampler = noisy_label_indices, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
                    new_clean_sample_indices = BALDDropout(noisy_trainloader,model1,noisy_label_indices,args.round_budget,device,args)
                    cleaned_sample_indices.extend(new_clean_sample_indices)

                elif args.active_selection_method == "coreset_sampling":
                    
                    coreset_trainloader = FastDataLoader(train_dataset, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)
                    new_clean_sample_indices = Coreset(coreset_trainloader,model1,cleaned_sample_indices,noisy_label_indices,args.round_budget,device,args)
                    cleaned_sample_indices.extend(new_clean_sample_indices)

                
            if args.label_cleaning:
                clean_labels(train_dataset.targets, correct_labels,cleaned_sample_indices)

            
            if args.method == "cross_entropy" and not args.active_learning_only:

                # create a new dataloader with the original dataset after cleaning some labels
                clean_trainloader = FastDataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, num_workers=2, persistent_workers= True)

            else:
            
                # create a new dataloader after getting the new samples relabeled 
                sampler = data.sampler.SubsetRandomSampler(cleaned_sample_indices)
                clean_trainloader = FastDataLoader(train_dataset, sampler = sampler, batch_size=args.batch_size, shuffle = False, num_workers=2, persistent_workers= True)

    

            ####################### start the label cleaning round #################################################
            for epoch in range(args.recalibration_epochs):

                ### train with the clean samples
                train_top1, train_top5, train_loss, indexes, all_losses, targets = train(clean_trainloader, model1, avg_criterion, optimizer, epoch, device, args, corrupt_indices, uncorrupt_indices)
                # infer on validation set
                val_acc1, val_acc5, val_loss, true_targets_val, predicted_targets_val, val_probs = inference(valloader, model1, avg_criterion, device, args)
                # infer on test set
                test_acc1, test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs = inference(testloader, model1, avg_criterion, device, args)

                scheduler.step()

                # evaluate metrices on validation set
                val_precision, val_recall, val_f1score, val_avg_acc, val_auroc, val_mcc, _ = evaluate_and_log_metrices(val_acc1, \
                val_acc5, val_loss, true_targets_val, predicted_targets_val, val_probs, mode = "val")

                # evaluate metrices on test set
                test_precision, test_recall, test_f1score, test_avg_acc, test_auroc, test_mcc, class_report_pdf_test = evaluate_and_log_metrices(test_acc1, \
                test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs, mode = "test")
                
                # use the F1 score in the validation set to select the best model
                is_best = val_f1score > best_F1score_val
                if is_best:
                    if args.pretrained_model is not None:
                        experiment_name = args.dataset+"/"+args.arch+"/"+args.pretrained_model+"/"+"corrupt_prob_"+str(args.corrupt_prob)
                    else:
                        experiment_name = args.dataset+"/"+args.arch+"/"+"corrupt_prob_"+str(args.corrupt_prob)

                    ## only save the best model 
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model1.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()
                    }, is_best, experiment_name = experiment_name,filename= "checkpoint_best.pth.tar")

                    # update the test score obtained on the best validation set
                    test_best_acc1 = test_acc1
                    test_best_precision = test_precision
                    test_best_recall = test_recall
                    test_best_F1score = test_f1score
                    test_best_balanced_accuracy = test_avg_acc
                    test_best_AUROC = test_auroc
                    test_best_mcc = test_mcc

                # update all best scores on the validation set, used for selecting the best model
                best_precision_val = max(best_precision_val,val_precision)
                best_recall_val = max(best_recall_val,val_recall)
                best_F1score_val = max(best_F1score_val,val_f1score)
                best_balanced_accuracy_val = max(best_balanced_accuracy_val,val_avg_acc)
                best_AUROC_val = max(best_AUROC_val,val_auroc)
                best_mcc_val = max(best_mcc_val,val_mcc)

                # log train statistics, val and test statistics are logged in "evaluate_and_log_metrices()"
                run["train/loss_model1"].log(train_loss)
                run["train/top1_model1"].log(train_top1)
                run["train/top2_model1"].log(train_top5)
                    

                # compute the average of the last 5 epochs
                if (epoch - args.recalibration_epochs) <=5: 
                    last_accuracies.append(test_acc1.cpu().numpy())
                    last_precisions.append(test_precision)
                    last_recalls.append(test_recall)
                    last_F1scores.append(test_f1score)
                    last_balanced_accuracies.append(test_avg_acc)
                    last_AUROC.append(test_auroc)
                    last_mcc.append(test_mcc)

            # log the best and the average of last 5 epochs
            run["test/last_5_avg_acc"].log(np.mean(last_accuracies)) 
            run["test/top1_best_acc"].log(test_best_acc1)
            run["test/last_5_avg_precision"].log(np.mean(last_precisions)) 
            run["test/top1_best_precision"].log(test_best_precision)
            run["test/last_5_avg_recall"].log(np.mean(last_recalls)) 
            run["test/top1_best_recall"].log(test_best_recall)
            run["test/last_5_avg_f1score"].log(np.mean(last_F1scores)) 
            run["test/top1_best_f1score"].log(test_best_F1score)
            run["test/last_5_avg_AUROC"].log(np.mean(last_AUROC)) 
            run["test/top1_best_AUROC"].log(test_best_AUROC)
            run["test/last_5_avg_balanced_acc"].log(np.mean(last_balanced_accuracies)) 
            run["test/top1_best_balanced_acc"].log(test_best_balanced_accuracy)
            run["test/last_5_avg_MCC"].log(np.mean(last_mcc)) 
            run["test/top1_best_MCC"].log(test_best_mcc)

            # log the current budget
            run["round_budgets"].log(currrent_budget)


        # if round is 0, we don't train the model only evaluate existing method 
        else:
            
            # no need to train, only infer and evaluate on the test dataset
            test_acc1, test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs = inference(testloader, model1, avg_criterion, device, args)
            test_precision, test_recall, test_f1score, test_avg_acc, test_auroc, test_mcc, class_report_pdf_test = evaluate_and_log_metrices(test_acc1, \
            test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs, mode = "test")
        
            # save the model
            if args.pretrained_model is not None:
                experiment_name = args.dataset+"/"+args.arch+"/"+args.pretrained_model+"/"+"corrupt_prob_"+str(args.corrupt_prob)
            else:
                experiment_name = args.dataset+"/"+args.arch+"/"+"corrupt_prob_"+str(args.corrupt_prob)

            save_checkpoint({
                'epoch': 0,
                'arch': args.arch,
                'state_dict': model1.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best=True, experiment_name = experiment_name,filename= "checkpoint_best.pth.tar")


            # save the best scores 
            run["test/top1_best_acc"].log(test_acc1)
            run["test/top1_best_precision"].log(test_precision)
            run["test/top1_best_recall"].log(test_recall)
            run["test/top1_best_f1score"].log(test_f1score)
            run["test/top1_best_AUROC"].log(test_auroc)
            run["test/top1_best_balanced_acc"].log(test_avg_acc)
            run["test/top1_best_MCC"].log(test_mcc)

            # log the current_budget
            run["round_budgets"].log(currrent_budget)


        currrent_budget += args.round_budget

        ################ additional reports from the last epoch ###############

        # save the whole test report of the last epoch
        run["classification_report"].upload(File.as_html(class_report_pdf_test))

        # compute the softmax probability density for each classes in test set
        all_class_prob = []
        for true in range(NUM_CLASSES):
            ind = np.where(np.array(true_targets_test) == true)[0]
            class_probs = test_probs[ind]
            class_probs_avg = np.mean(class_probs, axis = 0)
            all_class_prob.append(class_probs_avg)


        # plot the probability density 
        for cls_prob in all_class_prob:
            fig, ax = plt.subplots()
            plt.bar(range(len(cls_prob)),cls_prob, align = "center")
            plt.xticks(range(len(cls_prob)))
            plt.grid()
            run["Probability map"].log(fig)
            plt.close()

        # save the confusion matrix for the last epoch
        print ("Saving Confusion Matrix")
        fig,ax= utils.plot_confusion_matrix(true_targets_test,predicted_targets_test, [str(c) for c in range (NUM_CLASSES)],title='Confusion Matrix')
        fig.savefig(os.path.join(args.save_dir, args.dataset,experiment_name ,args.confusion_matrix))
        run["confusion_matrix"].upload(File.as_image(fig))
        plt.close()

        run.sync(wait=True)

def evaluate_and_log_metrices(acc1, acc5, loss, true_targets, predicted_targets, probs, mode = "test"):

    # get the score for test set 
    per_class_acc, avg_acc, overall_acc= utils.get_acc(true_labels= true_targets, predicted_labels= predicted_targets)
    class_report_pdf = utils.class_report(true_targets,predicted_targets, probs)
    precision = class_report_pdf.loc["macro avg","precision"]
    recall = class_report_pdf.loc["macro avg","recall"]
    f1score = class_report_pdf.loc["macro avg","f1-score"]
    auroc = roc_auc_score(true_targets,probs, multi_class= "ovr")
    mcc_value = mcc(true_targets,predicted_targets)

    # compute per class statistics
    per_class_recall = class_report_pdf["recall"].values.tolist()
    per_class_precision = class_report_pdf["precision"].values.tolist()
    per_class_f1_score = class_report_pdf["f1-score"].values.tolist()


    ################### Log statistics to Neputne.ai #######################

    if mode == "test":
        run["test/loss"].log(loss)
        run["test/top1"].log(acc1)
        run["test/top5"].log(acc5)
        run["test/Precision"].log(precision)
        run["test/Recall"].log(recall)
        run["test/F1-Score"].log(f1score)
        run["test/Avg Acc"].log(avg_acc)
        run["test/AUROC"].log(auroc)
        run["test/MCC"].log(mcc_value)

        # log per-class (accuracy, precision score, recall, and F1 score)
        all_class_acc = []
        all_class_precision = []
        all_class_recall = []
        all_class_f1score = []

        for cls in range(len(per_class_acc)):
            run["test/Class Acc: "+ str(cls)].log(per_class_acc[cls])
            run["test/Class Precision: "+ str(cls)].log(per_class_precision[cls])
            run["test/Class Recall: "+ str(cls)].log(per_class_recall[cls])
            run["test/Class F1score: "+ str(cls)].log(per_class_f1_score[cls])

            # append per-class values to a list (used for barplots below)
            all_class_acc.append(per_class_acc[cls])
            all_class_precision.append(per_class_precision[cls])
            all_class_recall.append(per_class_recall[cls])
            all_class_f1score.append(per_class_f1_score[cls])

            
        ############################# log barplots ############################
        # log per-class accuracy barplot
        fig, ax = plt.subplots()
        plt.bar(range(cls+1),all_class_acc, align = "center")
        plt.xticks(range(cls+1))
        plt.grid()
        run["Per class Acc"].log(fig)
        plt.close()

        # log per-class precision barplot
        fig, ax = plt.subplots()
        plt.bar(range(cls+1),all_class_precision, align = "center")
        plt.xticks(range(cls+1))
        plt.grid()
        run["Per class Precision"].log(fig)
        plt.close()

        # log per-class recall barplot
        fig, ax = plt.subplots()
        plt.bar(range(cls+1),all_class_recall, align = "center")
        plt.xticks(range(cls+1))
        plt.grid()
        run["Per class Recall"].log(fig)
        plt.close()

        # log per-class f1score barplot
        fig, ax = plt.subplots()
        plt.bar(range(cls+1),all_class_f1score, align = "center")
        plt.xticks(range(cls+1))
        plt.grid()
        run["Per class F1-score"].log(fig)
        plt.close()
    

    elif mode == "val":

        run["val/loss"].log(loss)
        run["val/top1"].log(acc1)
        run["val/top5"].log(acc5)
        run["val/Precision"].log(precision)
        run["val/Recall"].log(recall)
        run["val/F1-Score"].log(f1score)
        run["val/Avg Acc"].log(avg_acc)
        run["val/AUROC"].log(auroc)
        run["val/MCC"].log(mcc_value)


    return  precision, recall, f1score, avg_acc, auroc, mcc_value, class_report_pdf

        

def save_checkpoint(state, is_best, experiment_name, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.join(args.save_dir, args.dataset, experiment_name)):
        os.makedirs(os.path.join(args.save_dir, args.dataset, experiment_name))
    torch.save(state, os.path.join(args.save_dir, args.dataset, experiment_name, filename))
   

def random_corrupt_labels(targets, num_classes, corrupt_prob):
    corrupt_indices = []
    uncorrupt_indices = []
    all_targets = [i for i in range(num_classes)]
    for i in range(len(targets)): 
        if (random.random() <= corrupt_prob):
            corrupt_targets = all_targets.copy()
            corrupt_targets.remove(targets[i])
            rand = random.choice(corrupt_targets)
            corrupt_indices.append(i)
            targets[i] = rand
        else:
            uncorrupt_indices.append(i)

    return np.array(corrupt_indices), np.array(uncorrupt_indices)


def corrupt_labels(targets, num_classes, corrupt_prob, uncorrupt_list, args):

    file = args.dataset+"_corrupt_prob_"+str(args.corrupt_prob)+"_uncorrupted_classes_"+''.join(map(str,uncorrupt_list))+'.npy'
    if os.path.isfile(os.path.join("corrupted_labels", file)):
        data = np.load(os.path.join("corrupted_labels", file), allow_pickle = True)
        print ("loaded from previously save corrupted labels")
        labels = data.item().get("labels")
        uncorrupt_indices = data.item().get("uncorrupted_indices")
        corrupt_indices = data.item().get("corrupted_indices")
        
        for i in range(len(targets)): 
            targets[i] = labels[i]
    else:
        corrupt_indices = []
        uncorrupt_indices = []
        corrupted_targets = []
        all_targets = [i for i in range(num_classes)]
        for i in range(len(targets)): 
            if (random.random() <= corrupt_prob) and (targets[i] not in uncorrupt_list):
                corrupt_targets = list(filter(lambda x: x not in uncorrupt_list, all_targets))
                corrupt_targets.remove(targets[i])
                rand = random.choice(corrupt_targets) 
                targets[i] = rand
                corrupt_indices.append(i)
                corrupted_targets.append(rand)
            else:
                uncorrupt_indices.append(i)
        print ("generated new corrupted labels")
        ## also save the labels and indices
        data = {"labels":targets,"uncorrupted_indices":np.array(uncorrupt_indices), "corrupted_indices":np.array(corrupt_indices)}
        if not os.path.exists('corrupted_labels'):
            os.makedirs('corrupted_labels')
        np.save(os.path.join("corrupted_labels", file), data, allow_pickle = True)
        corrupt_indices = np.array(corrupt_indices)
        uncorrupt_indices = np.array(uncorrupt_indices)
        
    return corrupt_indices, uncorrupt_indices


def clean_labels(noisy_targets, correct_targets, cleaned_sample_indices):

    for i in range (len(noisy_targets)):
        if i in cleaned_sample_indices:
            noisy_targets[i] = correct_targets[i]


def corrupt_labels_class_dependent(targets, num_classes, corrupt_prob, candidate_classes, args):

    file = args.dataset+"_class_dependent_"+"_corrupt_prob_"+str(args.corrupt_prob)+'.npy'
    if os.path.isfile(os.path.join("corrupted_labels", file)):
        data = np.load(os.path.join("corrupted_labels", file), allow_pickle = True)
        print ("loaded from previously save corrupted labels")
        labels = data.item().get("labels")
        uncorrupt_indices = data.item().get("uncorrupted_indices")
        corrupt_indices = data.item().get("corrupted_indices")
        
        for i in range(len(targets)): 
            targets[i] = labels[i]
    else:
        corrupt_indices = []
        uncorrupt_indices = []
        corrupted_targets = []
        all_targets = [i for i in range(num_classes)]
        for i in range(len(targets)): 
            if random.random() <= corrupt_prob:
                corrupt_targets = list(filter(lambda x: x in candidate_classes[targets[i]], all_targets))
                corrupt_targets.remove(targets[i])
                if len(corrupt_targets)>0:
                    rand = random.choice(corrupt_targets) 
                    targets[i] = rand
                    corrupt_indices.append(i)
                    corrupted_targets.append(rand)
                else:
                    uncorrupt_indices.append(i)
            else:
                uncorrupt_indices.append(i)
        print ("generated new corrupted labels")
        ## also save the labels and indices
        data = {"labels":targets,"uncorrupted_indices":np.array(uncorrupt_indices), "corrupted_indices":np.array(corrupt_indices)}
        if not os.path.exists('corrupted_labels'):
            os.makedirs('corrupted_labels')
        np.save(os.path.join("corrupted_labels", file), data, allow_pickle = True)
        corrupt_indices = np.array(corrupt_indices)
        uncorrupt_indices = np.array(uncorrupt_indices)
        
    return corrupt_indices, uncorrupt_indices

   
if __name__ == "__main__":

    run = neptune.init_run(
    project="Project-Name",
    api_token="Your API token",
)  # your credentials


    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    parser.add_argument('--root', metavar='DIR', default='data/',
                        help='path ot the dataset')
    parser.add_argument('--method', default='coteaching', type= str, choices = ["coteaching","coteaching_VOG","cross_entropy","scratch"])

    #### co-teaching parameters
    parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
    parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')

    parser.add_argument('--save_dir', default='checkpoints', type= str)
    parser.add_argument('-a', '--arch', type=str, default='resnet18',
                        help='model architecture you want ti use')
    parser.add_argument('--dataset', default = "cifar10", type = str, choices = ["cifar10","imbalanced_cifar10", "isic","drd","kvasir","imbalanced_histopathology"])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    
    # choose different learning rates 
    parser.add_argument('--lr_LNL', default=0.01, type=float,help='initial learning rate for LNL')
    parser.add_argument('--lr_AL', default=0.01, type=float, 
                        help='initial learning rate for finetuning on active learning')
    

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=5, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--use_gpu', default=True, type= bool,
                        help='use the default GPU')
    parser.add_argument('--gpu_id', type =str , default= '0',
                        help='select a gpu id')
    parser.add_argument('--confusion_matrix', default='confusion.png',
                        help='filename of the confusion matrix')
    parser.add_argument('--pretrained_model', type =str , default= None, help='select if to use pretrained model or not?')


    # VOG specific parameter 
    parser.add_argument('--all_activations', default=0, type= int, help = "compute gradients of all softmax activations w.r.t inputs")
    parser.add_argument('--mix_ratio', type= float, default= 0.2,
                        help='mix ratio for loss and VOG selection, 0.2 means 0.2 * VOG selection + 0.8 * loss selection')
    
    
    # related to label corruption
    parser.add_argument('--corrupt_prob', type= float, default=1.1,
                        help='probability by which the labels are corrupted')
    parser.add_argument('--uncorrupt_classes', required= True, type =str, help = "list of classes that you don't want to corrupt")

    
    ### LNL epochs
    parser.add_argument('--LNL_epochs', default=50, type=int,help='epochs upto which to train the LNL method')
    ### cross_entropy epochs
    parser.add_argument('--CE_epochs', default=50, type=int,help='epochs upto which to train the vanilla cross entropy method')


    ## active label cleaning 
    parser.add_argument('--label_cleaning_rounds', default=5, type=int,help='number of rounds to reach the budget, i.e at each recalibration epochs budget*(1/label_cleaning_rounds) of examples are relabeled')
    parser.add_argument('--round_budget', default=100, type=int,help='increase the annotation budget by')
    parser.add_argument('--recalibration_epochs', default=20, type=int,help='number of epochs to train on cleaned examples before another phase of active label cleaning')
    parser.add_argument('--active_selection_method',default="random", type=str,help='select the active learning method', choices=["random","all_random","entropy_sampling","least_confidence_sampling","margin_sampling","BALD_sampling","coreset_sampling"])
    parser.add_argument('--use_LNL_cleaned_labels', type=str2bool, default = False, help='use the clean labels guessed by LNL as true labels')           
    parser.add_argument('--label_cleaning', type=str2bool , default= True, help='Clean the labels or not, it is False when using on Active learning with label cleaning')
    parser.add_argument('--active_learning_only', type=str2bool , default= False, help='Dont use noisy labels at all, start relableing from scratch (similar to Active Learning Only)')
    parser.add_argument('--scratch_after_LNL', type=str2bool , default= False, help='Restart to train from scratch using the labels guessed as clean by LNL')


    args = parser.parse_args()
    params = vars(args)
    run["parameters"] = params
    run["all files"].upload_files("*.py")

    ############parameters before main ########################

    # parse some arguments 
    uncorrupt_classes = [int(item) for item in args.uncorrupt_classes.split(',')]
   
    if args.use_gpu:
        device = 'cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu'

    fix_seed(args.seed)
    main(args)
