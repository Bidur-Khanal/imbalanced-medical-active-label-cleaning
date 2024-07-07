
import torch
from coreset import Coreset_Greedy
import torch.nn.functional as F


# Most of the Active learning codes here are adopted from https://github.com/ej0cl6/deep-active-learning/tree/master

def predict_prob(dataloader, model, device,args, return_targets = None):
    probs = []
    targets = []
    def run_inference(loader):
        with torch.no_grad():
            for i, (images, target, index) in enumerate(loader):
                if args.use_gpu:
                    images = images.to(device)
                    target = target.to(device)
            
                # compute output
                output = model(images)
                prob = torch.nn.functional.softmax(output, dim=1)
                probs.append(prob.cpu())
                targets.append(target.cpu())
                
    
    # switch to evaluate mode
    model.eval()
    run_inference(dataloader)

    if return_targets:
        return torch.cat(probs, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(probs, dim=0)


def predict_prob_dropout_split(dataloader, model, device,args, n_drop=10):
        
    probs = []
    def run_inference(loader):
        for i in range(n_drop):
            with torch.no_grad():
                for j, (images, target, index) in enumerate(loader):
                    if args.use_gpu:
                        images = images.to(device)
                        target = target.to(device)
                
                    # compute output
                    output = model(images)
                    probs.append(torch.nn.functional.softmax(output, dim=1).cpu())
              
    # switch to evaluate mode
    model.train()
    run_inference(dataloader)
    probs = torch.cat(probs, dim=0)
    return probs.view(n_drop,-1,probs.shape[1])


def get_embeddings(dataloader, model,device,args, layer_name='avgpool'):
    
    model.eval()
    features = []

    # Define the hook function
    def hook_function(module, input, output):
        # Append the output to the features list
        features.append(output.detach())

    # Register the hook to the specified layer
    layer = dict([*model.named_children()])[layer_name]
    hook = layer.register_forward_hook(hook_function)
    

    with torch.no_grad():
        for i, (images, target, index) in enumerate(dataloader):
            if args.use_gpu:
                images = images.to(device)
            model(images)

    hook.remove()
    
    # Reshape and concatenate all collected features
    features = torch.cat(features,dim=0).view(-1,features[0].shape[1])

    return features.cpu()


def EntropySampling(dataloader, model, noisy_label_indices, budget,device,  args):

        
        probs = predict_prob(dataloader, model, device,args)
        epsilon = 1e-10
        log_probs = torch.log(probs+epsilon)
        uncertainties = (probs*log_probs).sum(1)
        return (torch.tensor(noisy_label_indices)[uncertainties.sort()[1][:budget]]).tolist()


def LeastConfidence(dataloader, model, noisy_label_indices, budget,device, args):
    
        probs = predict_prob(dataloader, model, device,args)
        uncertainties = probs.max(1)[0]
        return (torch.tensor(noisy_label_indices)[uncertainties.sort()[1][:budget]]).tolist()



def MarginSampling(dataloader, model, noisy_label_indices, budget,device, args):
    
        probs = predict_prob(dataloader, model, device,args)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return (torch.tensor(noisy_label_indices)[uncertainties.sort()[1][:budget]]).tolist()


def BALDDropout(dataloader, model, noisy_label_indices, budget,device, args):
    
        probs = predict_prob_dropout_split(dataloader, model, device,args)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return (torch.tensor(noisy_label_indices)[uncertainties.sort()[1][:budget]]).tolist()


def Coreset(dataloader, model, clean_label_indices, noisy_label_indices, budget,device, args):
        
        embeddings = get_embeddings(dataloader, model,device,args)
        coreset = Coreset_Greedy(embeddings)
        new_batch, max_distance = coreset.sample(clean_label_indices, budget)
        
        return new_batch


def labels_to_one_hot(labels, num_classes):
    """
    Convert class label tensor to one-hot encoded format.

    :param labels: A tensor of class labels. Shape: (num_samples,)
    :param num_classes: The total number of classes.
    :return: A tensor in one-hot encoded format. Shape: (num_samples, num_classes)
    """
    # Ensure labels are of type torch.long
    labels = labels.to(torch.long)

    return F.one_hot(labels, num_classes=num_classes)


def cross_entropy(predicted_distribution: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalised cross-entropy between the predicted and target distributions using PyTorch.
    :param predicted_distribution: Predicted distribution, shape = (num_samples, num_classes)
    :param target_distribution: Target distribution, shape = (num_samples, num_classes)
    :return: The cross-entropy for each sample as a PyTorch tensor
    """
    num_classes = predicted_distribution.shape[1]
    if target_distribution.dim() == 1:
        target_distribution = labels_to_one_hot(target_distribution,num_classes)
    return -torch.sum(target_distribution * torch.log(predicted_distribution + 1e-12) / torch.log(torch.tensor(num_classes, dtype=torch.float)), dim=-1)
