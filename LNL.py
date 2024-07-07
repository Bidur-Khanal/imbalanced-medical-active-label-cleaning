import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from solver import  inference_coteaching_general, train_coteaching_general,train_coteaching_simplified_mix_VOG, inference, compute_gradient_for_data, train
import neptune as neptune
import matplotlib.pyplot as plt
from neptune.types import File
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef as mcc
import utils 


class CoteachingLNL:

    def __init__(self, model1, model2, device, args):

        self.args = args
        self.device = device

        # dual models for LNL
        self.model1 = model1
        self.model2 = model2

        # compute per sample loss, use reduction = "none"
        self.criterion  = nn.CrossEntropyLoss(reduction = "none").to(device)

        # compute loss per batch
        self.criterion_mean  = nn.CrossEntropyLoss().to(device)

        self.optimizer1 = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model1.parameters()), self.args.lr_LNL,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        self.optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model2.parameters()), self.args.lr_LNL,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=self.args.LNL_epochs)
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=self.args.LNL_epochs)

        # LNL coteaching hyperparameters
        if self.args.forget_rate is None:
            self.forget_rate=self.args.corrupt_prob
        else:
            self.forget_rate=self.args.forget_rate



    def train(self,trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices, original_targets, num_classes, run):
    
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

        rate_schedule = np.ones(self.args.epochs)*self.forget_rate
        rate_schedule[:self.args.num_gradual] = np.linspace(0, self.forget_rate**self.args.exponent, self.args.num_gradual)

        for epoch in range(self.args.LNL_epochs):
         
            # train for one epoch
            train1_top1, train1_top5, train1_loss, train2_top1, train2_top5, train2_loss, _, \
            _, indexes, guessed_uncorrupted, all_losses1, targets = train_coteaching_general(trainloader, \
            self.model1, self.model2, self.criterion, self.optimizer1, self.optimizer2, rate_schedule,epoch, \
            self.device, self.args, corrupt_indices, uncorrupt_indices)

            # evaluate on validation set
            val_acc1, val_acc5, val_loss, true_targets_val, predicted_targets_val, val_probs = \
            inference_coteaching_general(valloader, self.model1, self.model2, self.criterion, self.device, self.args)


            # evaluate on test set
            test_acc1, test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs = \
            inference_coteaching_general(testloader, self.model1, self.model2, self.criterion, self.device, self.args)
            
            
            self.scheduler1.step()
            self.scheduler2.step()

            ################################# Evaluation metrices ###########################
            # get the score for validation set
            val_per_class_acc, val_avg_acc, val_overall_acc= utils.get_acc(true_labels= true_targets_val, predicted_labels= predicted_targets_val)
            class_report_pdf_val = utils.class_report(true_targets_val,predicted_targets_val, val_probs)
            val_precision = class_report_pdf_val.loc["macro avg","precision"]
            val_recall = class_report_pdf_val.loc["macro avg","recall"]
            val_f1score = class_report_pdf_val.loc["macro avg","f1-score"]
            val_auroc = roc_auc_score(true_targets_val,val_probs, multi_class= "ovr")
            val_mcc = mcc(true_targets_val,predicted_targets_val)


            # get the score for test set 
            test_per_class_acc, test_avg_acc, test_overall_acc= utils.get_acc(true_labels= true_targets_test, predicted_labels= predicted_targets_test)
            class_report_pdf_test = utils.class_report(true_targets_test,predicted_targets_test, test_probs)
            test_precision = class_report_pdf_test.loc["macro avg","precision"]
            test_recall = class_report_pdf_test.loc["macro avg","recall"]
            test_f1score = class_report_pdf_test.loc["macro avg","f1-score"]
            test_auroc = roc_auc_score(true_targets_test,test_probs, multi_class= "ovr")
            test_mcc = mcc(true_targets_test,predicted_targets_test)


            # use the F1 score in the validation set to select the best model
            is_best = val_f1score > best_F1score_val
            if is_best:

                ''' uncomment this if we want to save the LNL model'''
                # if self.args.pretrained_model is not None:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)
                # else:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)

                # #only save the best model 
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': self.args.arch,
                #     'state_dict': self.model1.state_dict(),
                #     'optimizer' : self.optimizer1.state_dict(),
                #     'scheduler' : self.scheduler1.state_dict()
                # }, is_best, experiment_name = Experiment_Name,filename= "checkpoint_best.pth.tar")

                test_best_acc1 = test_acc1
                test_best_precision = test_precision
                test_best_recall = test_recall
                test_best_F1score = test_f1score
                test_best_balanced_accuracy = test_avg_acc
                test_best_AUROC = test_auroc
                test_best_mcc = test_mcc

                

            ### update all the best metrices on the validation set, used for selecting the best model
            best_precision_val = max(best_precision_val,val_precision)
            best_recall_val = max(best_recall_val,val_recall)
            best_F1score_val = max(best_F1score_val,val_f1score)
            best_balanced_accuracy_val = max(best_balanced_accuracy_val,val_avg_acc)
            best_AUROC_val = max(best_AUROC_val,val_auroc)
            best_mcc_val = max(best_mcc_val,val_mcc)



            per_class_recall = class_report_pdf_test["recall"].values.tolist()
            per_class_precision = class_report_pdf_test["precision"].values.tolist()
            per_class_f1_score = class_report_pdf_test["f1-score"].values.tolist()

            # send the per-class (accuracy, precision score, recall, and F1 score) of test dataset to neptune
            all_class_acc = []
            all_class_precision = []
            all_class_recall = []
            all_class_f1score = []

            for cls in range(len(test_per_class_acc)):
                run["LNL/test/Class Acc: "+ str(cls)].log(test_per_class_acc[cls])
                run["LNL/test/Class Precision: "+ str(cls)].log(per_class_precision[cls])
                run["LNL/test/Class Recall: "+ str(cls)].log(per_class_recall[cls])
                run["LNL/test/Class F1score: "+ str(cls)].log(per_class_f1_score[cls])

                all_class_acc.append(test_per_class_acc[cls])
                all_class_precision.append(per_class_precision[cls])
                all_class_recall.append(per_class_recall[cls])
                all_class_f1score.append(per_class_f1_score[cls])

            

            ################## plot per class accuracy, precision score, recall, and F1 score of test dataset in barplot format ################
            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_acc, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Acc"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_precision, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Precision"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_recall, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Recall"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_f1score, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class F1-score"].log(fig)
            plt.close()
            ################## plot per class accuracy, precision score, recall, and F1 score of test dataset in barplot format ################


        
            #### save the values to neptune #######
            run["LNL/train/loss_model1"].log(train1_loss)
            run["LNL/train/loss_model2"].log(train2_loss)

            run["LNL/train/top1_model1"].log(train1_top1)
            run["LNL/train/top1_model2"].log(train2_top1)
            run["LNL/train/top2_model1"].log(train1_top5)
            run["LNL/train/top2_model2"].log(train2_top5)

            run["LNL/val/loss"].log(val_loss)
            run["LNL/val/top1"].log(val_acc1)
            run["LNL/val/top5"].log(val_acc5)
            run["LNL/val/Precision"].log(val_precision)
            run["LNL/val/Recall"].log(val_recall)
            run["LNL/val/F1-Score"].log(val_f1score)
            run["LNL/val/Avg Acc"].log(val_avg_acc)
            run["LNL/val/AUROC"].log(val_auroc)
            run["LNL/val/MCC"].log(val_mcc)


            run["LNL/test/loss"].log(test_loss)
            run["LNL/test/top1"].log(test_acc1)
            run["LNL/test/top5"].log(test_acc5)
            run["LNL/test/Precision"].log(test_precision)
            run["LNL/test/Recall"].log(test_recall)
            run["LNL/test/F1-Score"].log(test_f1score)
            run["LNL/test/Avg Acc"].log(test_avg_acc)
            run["LNL/test/AUROC"].log(test_auroc)
            run["LNL/test/MCC"].log(test_mcc)


            #### the average of last 5 epochs
            if (epoch - self.args.LNL_epochs) <=5: 
                last_accuracies.append(test_acc1.cpu().numpy())
                last_precisions.append(test_precision)
                last_recalls.append(test_recall)
                last_F1scores.append(test_f1score)
                last_balanced_accuracies.append(test_avg_acc)
                last_AUROC.append(test_auroc)
                last_mcc.append(test_mcc)


            # the sample selection strategies eval
            percent_correct_guess = 100*(np.count_nonzero(np.in1d(guessed_uncorrupted, uncorrupt_indices)))/len(uncorrupt_indices)
            all_losses1 = np.array(all_losses1)[np.argsort(indexes)]
            
            ### class-wise eval of percentage guess
            all_class_percentage = []
            for t in np.unique(targets):
                class_idx = np.where(np.array(original_targets)[uncorrupt_indices] == t)[0]
                actual_class_idx = uncorrupt_indices[class_idx]
                predicted_class_percentage = 100*(np.count_nonzero(np.in1d(guessed_uncorrupted,actual_class_idx)))/len(actual_class_idx)
                all_class_percentage.append(predicted_class_percentage)
                run["LNL/Guessed from Class: "+str(t)].log(predicted_class_percentage)

            ################## plot the guess from each class ################
            fig, ax = plt.subplots()
            plt.bar(range(t+1),all_class_percentage, align = "center")
            plt.xticks(range(t+1))
            plt.grid()
            run["LNL/Class Guess"].log(fig)
            plt.close()
            

            ################ plot the  box plots ####################
            plt.rcParams["figure.figsize"] = [15, 4]
            fig, ax = plt.subplots()
            ax.boxplot([all_losses1[uncorrupt_indices],all_losses1[corrupt_indices], all_losses1[guessed_uncorrupted]], vert = False, labels = ["uncorrupted", "corrupted", "selected as uncorrupted"],patch_artist=True, notch = True)
            plt.legend()
            plt.grid()
            run["LNL/losses selection"].log(fig)
            plt.close()

            run["LNL/percentage_correct_guessed"].log(percent_correct_guess)

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

        # save the whole report of the last epoch 
        run["LNL/classification_report"].upload(File.as_html(class_report_pdf_test))


        all_class_prob = []
        # also compute the test softmax probabilities of given class for all the example
        for true in range(num_classes):
            ind = np.where(np.array(true_targets_test) == true)[0]
            class_probs = test_probs[ind]
            class_probs_avg = np.mean(class_probs, axis = 0)
            all_class_prob.append(class_probs_avg)



        # probability map only for the epoch
        for cls_prob in all_class_prob:
            fig, ax = plt.subplots()
            plt.bar(range(len(cls_prob)),cls_prob, align = "center")
            plt.xticks(range(len(cls_prob)))
            plt.grid()
            run["LNL/Probability map"].log(fig)
            plt.close()

        # save the confusion matrix for the last epoch
        print ("Saving Confusion Matrix")
        fig,ax= utils.plot_confusion_matrix(true_targets_test,predicted_targets_test, [str(c) for c in range (num_classes)],title='Confusion Matrix')
        # fig.savefig(os.path.join(self.args.save_dir, self.args.dataset,Experiment_Name ,self.args.confusion_matrix))
        plt.close()
        run["LNL/confusion_matrix"].upload(File.as_image(fig))


        # ############# save the last epoch model and the guessed clean label indices #################
        # if self.args.pretrained_model is not None:
        #     Experiment_Name = self.args.dataset+"/coteaching/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)
        # else:
        #     Experiment_Name = self.args.dataset+"/coteaching/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)

        # # only save the best model 
        # self.save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': self.args.arch,
        #     'state_dict': model1.state_dict(),
        #     'optimizer' : optimizer1.state_dict(),
        #     'scheduler' : scheduler1.state_dict()
        # }, is_best, experiment_name = Experiment_Name,filename= "checkpoint_epoch_"+str(self.args.LNL_epochs)+".pth.tar")

        # # save the guessed clean label indices 
        # self.save_guessed_clean_label_indices(guessed_uncorrupted,Experiment_Name)

        return guessed_uncorrupted


    def save_guessed_clean_label_indices(self,guessed_clean_indices, experiment_name):

        path = os.path.join(self.args.save_dir, self.args.dataset, experiment_name)
        if not os.path.exists(path):
                os.makedirs(path)
        file_name = path + "/LNL_correct_guessed_label_indices_epochs_"+str(self.args.LNL_epochs)+".npy"

        np.save(file_name, np.array(guessed_clean_indices,dtype='int'), allow_pickle = True)

    def save_checkpoint(self, state, is_best, experiment_name, filename='checkpoint.pth.tar'):
        if not os.path.exists(os.path.join(self.args.save_dir, self.args.dataset, experiment_name)):
            os.makedirs(os.path.join(self.args.save_dir, self.args.dataset, experiment_name))
        torch.save(state, os.path.join(self.args.save_dir, self.args.dataset, experiment_name, filename))
    


    def get_losses(self,all_losses, indexes, corrupt_indices, uncorrupt_indices):

        ###### corrupt and uncorrupt losses
        corrupt_indx = np.where(np.in1d(indexes, corrupt_indices))[0]
        uncorrupt_idx = np.where(np.in1d(indexes, uncorrupt_indices))[0]

        corrupt_all_losses = np.mean(all_losses[corrupt_indx])
        uncorrupt_all_losses = np.mean(all_losses[uncorrupt_idx])
        
        return corrupt_all_losses, uncorrupt_all_losses, np.mean(all_losses)
    
    

class CoteachingVOGLNL:

    def __init__(self, model1, model2, device, args):

        self.args = args
        self.device = device

        # dual models for LNL
        self.model1 = model1
        self.model2 = model2

        # compute per sample loss, use reduction = "none"
        self.criterion  = nn.CrossEntropyLoss(reduction = "none").to(device)
        # compute loss per batch
        self.criterion_mean  = nn.CrossEntropyLoss().to(device)

        self.optimizer1 = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model1.parameters()), self.args.lr_LNL,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        self.optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model2.parameters()), self.args.lr_LNL,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=self.args.LNL_epochs)
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=self.args.LNL_epochs)

        # LNL coteaching hyperparameters
        if self.args.forget_rate is None:
            self.forget_rate=self.args.corrupt_prob
        else:
            self.forget_rate=self.args.forget_rate



    def train(self,trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices, original_targets, num_classes, run):
    
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

        rate_schedule = np.ones(self.args.epochs)*self.forget_rate
        rate_schedule[:self.args.num_gradual] = np.linspace(0, self.forget_rate**self.args.exponent, self.args.num_gradual)

        for epoch in range(self.args.LNL_epochs):
         
            vog1, vog2 = self.get_VOG_saliency(epoch) 

            # train for one epoch
            train1_top1, train1_top5, train1_loss, train2_top1, train2_top5, train2_loss, indexes, guessed_uncorrupted, all_losses1,\
            targets = train_coteaching_simplified_mix_VOG(trainloader, self.model1, self.model2, self.criterion, self.optimizer1, self.optimizer2, \
            rate_schedule,epoch, self.device, self.args, vog1, vog2,corrupt_indices, uncorrupt_indices)
            
            # get the gradients 
            gradients1, gradient_indexes1 = compute_gradient_for_data(trainloader, self.model1, self.criterion_mean, self.optimizer1, self.device, self.args)
            gradients2, gradient_indexes2 = compute_gradient_for_data(trainloader, self.model2, self.criterion, self.optimizer2, self.device, self.args)
            
            
            # evaluate on validation set
            val_acc1, val_acc5, val_loss, true_targets_val, predicted_targets_val, val_probs = \
            inference_coteaching_general(valloader, self.model1, self.model2, self.criterion, self.device, self.args)


            # evaluate on test set
            test_acc1, test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs = \
            inference_coteaching_general(testloader, self.model1, self.model2, self.criterion, self.device, self.args)
            
                
            self.scheduler1.step()
            self.scheduler2.step()

            # save the saliency maps
            self.save_gradients(gradients1, gradients2, gradient_indexes1, gradient_indexes2, epoch)

            ################################# Evaluation metrices ###########################
            # get the score for validation set
            val_per_class_acc, val_avg_acc, val_overall_acc= utils.get_acc(true_labels= true_targets_val, predicted_labels= predicted_targets_val)
            class_report_pdf_val = utils.class_report(true_targets_val,predicted_targets_val, val_probs)
            val_precision = class_report_pdf_val.loc["macro avg","precision"]
            val_recall = class_report_pdf_val.loc["macro avg","recall"]
            val_f1score = class_report_pdf_val.loc["macro avg","f1-score"]
            val_auroc = roc_auc_score(true_targets_val,val_probs, multi_class= "ovr")
            val_mcc = mcc(true_targets_val,predicted_targets_val)


            # get the score for test set 
            test_per_class_acc, test_avg_acc, test_overall_acc= utils.get_acc(true_labels= true_targets_test, predicted_labels= predicted_targets_test)
            class_report_pdf_test = utils.class_report(true_targets_test,predicted_targets_test, test_probs)
            test_precision = class_report_pdf_test.loc["macro avg","precision"]
            test_recall = class_report_pdf_test.loc["macro avg","recall"]
            test_f1score = class_report_pdf_test.loc["macro avg","f1-score"]
            test_auroc = roc_auc_score(true_targets_test,test_probs, multi_class= "ovr")
            test_mcc = mcc(true_targets_test,predicted_targets_test)


            # use the F1 score in the validation set to select the best model
            is_best = val_f1score > best_F1score_val
            if is_best:

                ''' uncomment this if we want to save the LNL model'''
                # if self.args.pretrained_model is not None:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)
                # else:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)

                # ## only save the best model 
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': self.args.arch,
                #     'state_dict': self.model1.state_dict(),
                #     'optimizer' : self.optimizer1.state_dict(),
                #     'scheduler' : self.scheduler1.state_dict()
                # }, is_best, experiment_name = Experiment_Name,filename= "checkpoint_best.pth.tar")

                test_best_acc1 = test_acc1
                test_best_precision = test_precision
                test_best_recall = test_recall
                test_best_F1score = test_f1score
                test_best_balanced_accuracy = test_avg_acc
                test_best_AUROC = test_auroc
                test_best_mcc = test_mcc

                

            ### update all the best metrices on the validation set, used for selecting the best model
            best_precision_val = max(best_precision_val,val_precision)
            best_recall_val = max(best_recall_val,val_recall)
            best_F1score_val = max(best_F1score_val,val_f1score)
            best_balanced_accuracy_val = max(best_balanced_accuracy_val,val_avg_acc)
            best_AUROC_val = max(best_AUROC_val,val_auroc)
            best_mcc_val = max(best_mcc_val,val_mcc)



            per_class_recall = class_report_pdf_test["recall"].values.tolist()
            per_class_precision = class_report_pdf_test["precision"].values.tolist()
            per_class_f1_score = class_report_pdf_test["f1-score"].values.tolist()

            # send the per-class (accuracy, precision score, recall, and F1 score) of test dataset to neptune
            all_class_acc = []
            all_class_precision = []
            all_class_recall = []
            all_class_f1score = []

            for cls in range(len(test_per_class_acc)):
                run["LNL/test/Class Acc: "+ str(cls)].log(test_per_class_acc[cls])
                run["LNL/test/Class Precision: "+ str(cls)].log(per_class_precision[cls])
                run["LNL/test/Class Recall: "+ str(cls)].log(per_class_recall[cls])
                run["LNL/test/Class F1score: "+ str(cls)].log(per_class_f1_score[cls])

                all_class_acc.append(test_per_class_acc[cls])
                all_class_precision.append(per_class_precision[cls])
                all_class_recall.append(per_class_recall[cls])
                all_class_f1score.append(per_class_f1_score[cls])

            

            ################## plot per class accuracy, precision score, recall, and F1 score of test dataset in barplot format ################
            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_acc, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Acc"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_precision, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Precision"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_recall, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class Recall"].log(fig)
            plt.close()

            fig, ax = plt.subplots()
            plt.bar(range(cls+1),all_class_f1score, align = "center")
            plt.xticks(range(cls+1))
            plt.grid()
            run["LNL/Per class F1-score"].log(fig)
            plt.close()
            ################## plot per class accuracy, precision score, recall, and F1 score of test dataset in barplot format ################


        
            #### save the values to neptune #######
            run["LNL/train/loss_model1"].log(train1_loss)
            run["LNL/train/loss_model2"].log(train2_loss)

            run["LNL/train/top1_model1"].log(train1_top1)
            run["LNL/train/top1_model2"].log(train2_top1)
            run["LNL/train/top2_model1"].log(train1_top5)
            run["LNL/train/top2_model2"].log(train2_top5)

            run["LNL/val/loss"].log(val_loss)
            run["LNL/val/top1"].log(val_acc1)
            run["LNL/val/top5"].log(val_acc5)
            run["LNL/val/Precision"].log(val_precision)
            run["LNL/val/Recall"].log(val_recall)
            run["LNL/val/F1-Score"].log(val_f1score)
            run["LNL/val/Avg Acc"].log(val_avg_acc)
            run["LNL/val/AUROC"].log(val_auroc)
            run["LNL/val/MCC"].log(val_mcc)


            run["LNL/test/loss"].log(test_loss)
            run["LNL/test/top1"].log(test_acc1)
            run["LNL/test/top5"].log(test_acc5)
            run["LNL/test/Precision"].log(test_precision)
            run["LNL/test/Recall"].log(test_recall)
            run["LNL/test/F1-Score"].log(test_f1score)
            run["LNL/test/Avg Acc"].log(test_avg_acc)
            run["LNL/test/AUROC"].log(test_auroc)
            run["LNL/test/MCC"].log(test_mcc)


            #### the average of last 5 epochs
            if (epoch - self.args.LNL_epochs) <=5: 
                last_accuracies.append(test_acc1.cpu().numpy())
                last_precisions.append(test_precision)
                last_recalls.append(test_recall)
                last_F1scores.append(test_f1score)
                last_balanced_accuracies.append(test_avg_acc)
                last_AUROC.append(test_auroc)
                last_mcc.append(test_mcc)


            # the sample selection strategies eval
            percent_correct_guess = 100*(np.count_nonzero(np.in1d(guessed_uncorrupted, uncorrupt_indices)))/len(uncorrupt_indices)
            all_losses1 = np.array(all_losses1)[np.argsort(indexes)]
            
            ### class-wise eval of percentage guess
            all_class_percentage = []
            for t in np.unique(targets):
                class_idx = np.where(np.array(original_targets)[uncorrupt_indices] == t)[0]
                actual_class_idx = uncorrupt_indices[class_idx]
                predicted_class_percentage = 100*(np.count_nonzero(np.in1d(guessed_uncorrupted,actual_class_idx)))/len(actual_class_idx)
                all_class_percentage.append(predicted_class_percentage)
                run["LNL/Guessed from Class: "+str(t)].log(predicted_class_percentage)

            ################## plot the guess from each class ################
            fig, ax = plt.subplots()
            plt.bar(range(t+1),all_class_percentage, align = "center")
            plt.xticks(range(t+1))
            plt.grid()
            run["LNL/Class Guess"].log(fig)
            plt.close()
            

            ################ plot the  box plots ####################
            plt.rcParams["figure.figsize"] = [15, 4]
            fig, ax = plt.subplots()
            ax.boxplot([all_losses1[uncorrupt_indices],all_losses1[corrupt_indices], all_losses1[guessed_uncorrupted]], vert = False, labels = ["uncorrupted", "corrupted", "selected as uncorrupted"],patch_artist=True, notch = True)
            plt.legend()
            plt.grid()
            run["LNL/losses selection"].log(fig)
            plt.close()

            run["LNL/percentage_correct_guessed"].log(percent_correct_guess)

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

        # save the whole report of the last epoch 
        run["LNL/classification_report"].upload(File.as_html(class_report_pdf_test))


        all_class_prob = []
        # also compute the test softmax probabilities of given class for all the example
        for true in range(num_classes):
            ind = np.where(np.array(true_targets_test) == true)[0]
            class_probs = test_probs[ind]
            class_probs_avg = np.mean(class_probs, axis = 0)
            all_class_prob.append(class_probs_avg)



        # probability map only for the epoch
        for cls_prob in all_class_prob:
            fig, ax = plt.subplots()
            plt.bar(range(len(cls_prob)),cls_prob, align = "center")
            plt.xticks(range(len(cls_prob)))
            plt.grid()
            run["LNL/Probability map"].log(fig)
            plt.close()

        # save the confusion matrix for the last epoch
        print ("Saving Confusion Matrix")
        fig,ax= utils.plot_confusion_matrix(true_targets_test,predicted_targets_test, [str(c) for c in range (num_classes)],title='Confusion Matrix')
        # fig.savefig(os.path.join(self.args.save_dir, self.args.dataset,Experiment_Name ,self.args.confusion_matrix))
        plt.close()
        run["LNL/confusion_matrix"].upload(File.as_image(fig))

        # ############# save the last epoch model and the guessed clean label indices #################
        # if self.args.pretrained_model is not None:
        #     Experiment_Name = self.args.dataset+"/VOG/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)
        # else:
        #     Experiment_Name = self.args.dataset+"/VOG/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)

        # # only save the best model 
        # self.save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': self.args.arch,
        #     'state_dict': model1.state_dict(),
        #     'optimizer' : optimizer1.state_dict(),
        #     'scheduler' : scheduler1.state_dict()
        # }, is_best, experiment_name = Experiment_Name,filename= "checkpoint_epoch_"+str(self.args.LNL_epochs)+".pth.tar")

        # # save the guessed clean label indices 
        # self.save_guessed_clean_label_indices(guessed_uncorrupted,Experiment_Name)

        return guessed_uncorrupted


    def save_guessed_clean_label_indices(self,guessed_clean_indices, experiment_name):

        path = os.path.join(self.args.save_dir, self.args.dataset, experiment_name)
        if not os.path.exists(path):
                os.makedirs(path)
        file_name = path + "/LNL_correct_guessed_label_indices_epochs_"+str(self.args.LNL_epochs)+".npy"

        np.save(file_name, np.array(guessed_clean_indices,dtype='int'), allow_pickle = True)

        return guessed_uncorrupted

    def save_checkpoint(self, state, is_best, experiment_name, filename='checkpoint.pth.tar'):
        if not os.path.exists(os.path.join(self.args.save_dir, self.args.dataset, experiment_name)):
            os.makedirs(os.path.join(self.args.save_dir, self.args.dataset, experiment_name))
        torch.save(state, os.path.join(self.args.save_dir, self.args.dataset, experiment_name, filename))
    

    def save_gradients(self,gradients1,gradients2, gradient_indexes1, gradient_indexes2, epoch):

        if self.args.pretrained_model is not None:
            experiment_name = self.args.dataset+"/VOG/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)+"_mix_ratio_"+str(self.args.mix_ratio)
        else:
            experiment_name = self.args.dataset+"/VOG/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)+"_mix_ratio_"+str(self.args.mix_ratio)

        path = os.path.join(self.args.save_dir, self.args.dataset, experiment_name)
        if not os.path.exists(path):
                os.makedirs(path)
        file_name = path + "/weight_epochs_"+str(epoch)+".npy"
        
        gradients1 = gradients1[np.argsort(gradient_indexes1)]
        gradients2 = gradients2[np.argsort(gradient_indexes2)]

        data = {"gradients1":gradients1, "gradients2": gradients2}
        np.save(file_name, data, allow_pickle = True)


    def get_losses(self,all_losses, indexes, corrupt_indices, uncorrupt_indices):

        ###### corrupt and uncorrupt losses
        corrupt_indx = np.where(np.in1d(indexes, corrupt_indices))[0]
        uncorrupt_idx = np.where(np.in1d(indexes, uncorrupt_indices))[0]

        corrupt_all_losses = np.mean(all_losses[corrupt_indx])
        uncorrupt_all_losses = np.mean(all_losses[uncorrupt_idx])
        
        return corrupt_all_losses, uncorrupt_all_losses, np.mean(all_losses)
    
    def get_VOG_saliency(self,epoch, observe_epochs=5):

        if epoch >= observe_epochs:

            if self.args.pretrained_model is not None:
                experiment_name = self.args.dataset+"/VOG/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)+"_mix_ratio_"+str(self.args.mix_ratio)
            else:
                experiment_name = self.args.dataset+"/VOG/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)+"_mix_ratio_"+str(self.args.mix_ratio)
            path = os.path.join(self.args.save_dir, self.args.dataset, experiment_name)
            all_grads1 = []
            all_grads2 = []

            for i in range(observe_epochs):
                pre_epoch = (epoch-i-1)
                file_name = path + "/weight_epochs_"+str(pre_epoch)+".npy"
                data = np.load(file_name, allow_pickle = True)

                gradients1 = data.item().get("gradients1")
                gradients2 = data.item().get("gradients2")

                # all grads together
                all_grads1.append(gradients1)
                all_grads2.append(gradients2)

            # for all grads for model 1
            mean_grad1 = np.sum(np.array(all_grads1), axis=0)/len(all_grads1)
            checkpoint_vog1 = np.sqrt(np.sum([(mm-mean_grad1)**2 for mm in all_grads1], axis = 0)/len(all_grads1))
            image_vog1 = np.mean(checkpoint_vog1, axis = 1)


            # for all grads for model 2
            mean_grad2 = np.sum(np.array(all_grads2), axis=0)/len(all_grads2)
            checkpoint_vog2 = np.sqrt(np.sum([(mm-mean_grad2)**2 for mm in all_grads2], axis = 0)/len(all_grads2))
            image_vog2 = np.mean(checkpoint_vog2, axis = 1)

            #### delete the previous saliency weights to save space #####
            del_file_name = path + "/weight_epochs_"+str(epoch-observe_epochs)+".npy"
            os.remove(del_file_name)
            
        
            return image_vog1, image_vog2

        else:
            return None, None


class CrossEntropy:

    def __init__(self, model,device, args):

        self.args = args
        self.device = device
        self.model = model
        self.criterion  = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr_AL,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.CE_epochs)
        

      
    def train(self,trainloader, valloader, testloader, corrupt_indices, uncorrupt_indices, original_targets, num_classes, run):
    
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


        for epoch in range(self.args.CE_epochs):
        
            # train for one epoch
            train_top1, train_top5, train_loss, indexes, all_losses, targets = train(trainloader, self.model, \
            self.criterion, self.optimizer, epoch, self.device, self.args, corrupt_indices, uncorrupt_indices)

            # evaluate on validation set
            val_acc1, val_acc5, val_loss, true_targets_val, predicted_targets_val, val_probs = inference(valloader, \
            self.model, self.criterion, self.device, self.args)

            # evaluate on test set
            test_acc1, test_acc5, test_loss, true_targets_test, predicted_targets_test, test_probs = inference(testloader, \
            self.model, self.criterion, self.device, self.args)
        
            self.scheduler.step()

            ################################# Evaluation metrices ###########################
            # get the score for validation set
            val_per_class_acc, val_avg_acc, val_overall_acc= utils.get_acc(true_labels= true_targets_val, predicted_labels= predicted_targets_val)
            class_report_pdf_val = utils.class_report(true_targets_val,predicted_targets_val, val_probs)
            val_precision = class_report_pdf_val.loc["macro avg","precision"]
            val_recall = class_report_pdf_val.loc["macro avg","recall"]
            val_f1score = class_report_pdf_val.loc["macro avg","f1-score"]
            val_auroc = roc_auc_score(true_targets_val,val_probs, multi_class= "ovr")
            val_mcc = mcc(true_targets_val,predicted_targets_val)


            # get the score for test set 
            test_per_class_acc, test_avg_acc, test_overall_acc= utils.get_acc(true_labels= true_targets_test, predicted_labels= predicted_targets_test)
            class_report_pdf_test = utils.class_report(true_targets_test,predicted_targets_test, test_probs)
            test_precision = class_report_pdf_test.loc["macro avg","precision"]
            test_recall = class_report_pdf_test.loc["macro avg","recall"]
            test_f1score = class_report_pdf_test.loc["macro avg","f1-score"]
            test_auroc = roc_auc_score(true_targets_test,test_probs, multi_class= "ovr")
            test_mcc = mcc(true_targets_test,predicted_targets_test)



            # use the F1 score in the validation set to select the best model
            is_best = val_f1score > best_F1score_val
            if is_best:
            
                # if self.args.pretrained_model is not None:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+self.args.pretrained_model+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)
                # else:
                #     Experiment_Name = self.args.dataset+"/"+self.args.arch+"/"+"corrupt_prob_"+str(self.args.corrupt_prob)+"_seed_"+str(self.args.seed)

                # ## only save the best model 
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': self.args.arch,
                #     'state_dict': self.model.state_dict(),
                #     'optimizer' : self.optimizer.state_dict(),
                #     'scheduler' : self.scheduler.state_dict()
                # }, is_best, experiment_name = Experiment_Name,filename= "checkpoint_best.pth.tar")

                test_best_acc1 = test_acc1
                test_best_precision = test_precision
                test_best_recall = test_recall
                test_best_F1score = test_f1score
                test_best_balanced_accuracy = test_avg_acc
                test_best_AUROC = test_auroc
                test_best_mcc = test_mcc


            ### update all the best metrices on the validation set, used for selecting the best model
            best_precision_val = max(best_precision_val,val_precision)
            best_recall_val = max(best_recall_val,val_recall)
            best_F1score_val = max(best_F1score_val,val_f1score)
            best_balanced_accuracy_val = max(best_balanced_accuracy_val,val_avg_acc)
            best_AUROC_val = max(best_AUROC_val,val_auroc)
            best_mcc_val = max(best_mcc_val,val_mcc)


            per_class_recall = class_report_pdf_test["recall"].values.tolist()
            per_class_precision = class_report_pdf_test["precision"].values.tolist()
            per_class_f1_score = class_report_pdf_test["f1-score"].values.tolist()

            # send the per-class (accuracy, precision score, recall, and F1 score) of test dataset to neptune
            for cls in range(len(test_per_class_acc)):
                run["test/Class Acc: "+ str(cls)].log(test_per_class_acc[cls])
                run["test/Class Precision: "+ str(cls)].log(per_class_precision[cls])
                run["test/Class Recall: "+ str(cls)].log(per_class_recall[cls])
                run["test/Class F1score: "+ str(cls)].log(per_class_f1_score[cls])


             # get average train losses
            if self.args.corrupt_prob > 0. :
                corrupt_loss, uncorrupt_loss, avg_losses = self.get_losses(all_losses,indexes, corrupt_indices, uncorrupt_indices)
                run["train/all_losses"].log(avg_losses)
                run["train/corrupt_all_losses"].log(corrupt_loss)
                run["train/uncorrupt_all_losses"].log(uncorrupt_loss)

            run["train/loss"].log(train_loss)
            run["train/top1"].log(train_top1)
            run["train/top5"].log(train_top5)


            run["val/loss"].log(val_loss)
            run["val/top1"].log(val_acc1)
            run["val/top5"].log(val_acc5)
            run["val/Precision"].log(val_precision)
            run["val/Recall"].log(val_recall)
            run["val/F1-Score"].log(val_f1score)
            run["val/Avg Acc"].log(val_avg_acc)
            run["val/AUROC"].log(val_auroc)
            run["val/MCC"].log(val_mcc)


            run["test/loss"].log(test_loss)
            run["test/top1"].log(test_acc1)
            run["test/top5"].log(test_acc5)
            run["test/Precision"].log(test_precision)
            run["test/Recall"].log(test_recall)
            run["test/F1-Score"].log(test_f1score)
            run["test/Avg Acc"].log(test_avg_acc)
            run["test/AUROC"].log(test_auroc)
            run["test/MCC"].log(test_mcc)

            #### the average of last 5 epochs
            if (epoch - self.args.CE_epochs) <=5: 
                last_accuracies.append(test_acc1.cpu().numpy())
                last_precisions.append(test_precision)
                last_recalls.append(test_recall)
                last_F1scores.append(test_f1score)
                last_balanced_accuracies.append(test_avg_acc)
                last_AUROC.append(test_auroc)
                last_mcc.append(test_mcc)

        
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

        
        # save the whole report of the last epoch 
        run["classification_report"].upload(File.as_html(class_report_pdf_test))

        all_class_prob = []
        # also compute the test softmax probabilities of given class for all the example
        for true in range(num_classes):
            ind = np.where(np.array(true_targets_test) == true)[0]
            class_probs = test_probs[ind]
            class_probs_avg = np.mean(class_probs, axis = 0)
            all_class_prob.append(class_probs_avg)

        # probability map only for the epoch
        for cls_prob in all_class_prob:
            fig, ax = plt.subplots()
            plt.bar(range(len(cls_prob)),cls_prob, align = "center")
            plt.xticks(range(len(cls_prob)))
            plt.grid()
            run["Probability map"].log(fig)
            plt.close()

        # save the confusion matrix for the last epoch
        print ("Saving Confusion Matrix")
        fig,ax= utils.plot_confusion_matrix(true_targets_test,predicted_targets_test, [str(c) for c in range (num_classes)],title='Confusion Matrix')
        # fig.savefig(os.path.join(self.args.save_dir, self.args.dataset,Experiment_Name ,self.args.confusion_matrix))
        plt.close()
        run["confusion_matrix"].upload(File.as_image(fig))


    def save_checkpoint(self, state, is_best, experiment_name, filename='checkpoint.pth.tar'):
        if not os.path.exists(os.path.join(self.args.save_dir, self.args.dataset, experiment_name)):
            os.makedirs(os.path.join(self.args.save_dir, self.args.dataset, experiment_name))
        torch.save(state, os.path.join(self.args.save_dir, self.args.dataset, experiment_name, filename))


    def get_losses(self,all_losses, indexes, corrupt_indices, uncorrupt_indices):

        ###### corrupt and uncorrupt losses
        corrupt_indx = np.where(np.in1d(indexes, corrupt_indices))[0]
        uncorrupt_idx = np.where(np.in1d(indexes, uncorrupt_indices))[0]

        corrupt_all_losses = np.mean(all_losses[corrupt_indx])
        uncorrupt_all_losses = np.mean(all_losses[uncorrupt_idx])
        
        return corrupt_all_losses, uncorrupt_all_losses, np.mean(all_losses)