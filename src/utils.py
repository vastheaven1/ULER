import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from IPython.display import display, clear_output
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
np.random.seed(2024) 

# Save trained model to file
def save_model(args, model, name=''):
    if not os.path.exists(args.models_path):
        os.makedirs(args.models_path)
    torch.save(model, f'{args.models_path}/{name}.pt')
    
# Load saved model from file
def load_model(args, name=''):
    model = torch.load(f'{args.models_path}/{name}.pt')
    return model

# Save model predictions and true labels for evaluation
def save_label(args, test_predict_data,test_true_data, name=''):
    if not os.path.exists(args.labels_path):
        os.makedirs(args.labels_path)
    np.save(f'{args.labels_path}/{name + "_predict"}.npy',test_predict_data)
    np.save(f'{args.labels_path}/{name + "_true"}.npy',test_true_data)
    
# Load saved predictions and true labels
def load_label(args, name=''):
    test_predict_data = np.load(f'{args.labels_path}/{name + "_predict"}.npy')
    test_true_data = np.load(f'{args.labels_path}/{name + "_true"}.npy')
    return test_predict_data,test_true_data

# Save features from the last layer for visualization
def save_last_features(args, test_last_features,name=''):
    if not os.path.exists(args.last_features_path):
        os.makedirs(args.last_features_path)
    np.save(f'{args.last_features_path}/{name + "_features"}.npy',test_last_features)
    
# Load saved features from last layer   
def load_last_features(args, name=''):
    test_last_features = np.load(f'{args.last_features_path}/{name + "_features"}.npy')
    return test_last_features

# Save best training results to log file
def save_best_result_td(args,train_log_message,valid_log_message,test_log_message,name=''):
    if not os.path.exists(args.best_result_path):
        os.makedirs(args.best_result_path)
    best_result_path = f'{args.best_result_path}/{name}.txt'
    with open(best_result_path, 'a') as log_file:
        log_file.write(train_log_message + '\n')
        log_file.write(valid_log_message + '\n')
        log_file.write(test_log_message + '\n')
        
# Save best training results to log file        
def save_best_result_sd(args,train_log_message,test_log_message,name=''):
    if not os.path.exists(args.best_result_path):
        os.makedirs(args.best_result_path)
    best_result_path = f'{args.best_result_path}/{name}.txt'
    with open(best_result_path, 'a') as log_file:
        log_file.write(train_log_message + '\n')
        log_file.write(test_log_message + '\n')        
        
# Convert tensors to numpy arrays for visualization
def to_numpy(tensor_or_list):
    if isinstance(tensor_or_list, list):
        return [to_numpy(item) for item in tensor_or_list]
    elif torch.is_tensor(tensor_or_list):
        if tensor_or_list.is_cuda:
            return tensor_or_list.cpu().detach().numpy()
        else:
            return tensor_or_list.detach().numpy()
    else:
        return tensor_or_list
    
# The following functions are visualization functions for different tasks.
def save_confusion_matrix_valence2(hyp_params,test_predict_data,test_true_data,name):

    cm = confusion_matrix(test_true_data, test_predict_data) 
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Negative','Positive'], 
                yticklabels=['Negative','Positive']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_valence2(hyp_params,last_features,true_data,name):
    
    target_names = ['Negative','Positive']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10',2), edgecolor='none', s=30)
    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 2)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
    
def save_confusion_matrix_arousal2(hyp_params,test_predict_data,test_true_data,name):

    cm = confusion_matrix(test_true_data, test_predict_data) 
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Low','High'], 
                yticklabels=['Low','High']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_arousal2(hyp_params,last_features,true_data,name):
    
    target_names = ['Low','High']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10',2), edgecolor='none', s=30)
    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 2)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
    
def save_confusion_matrix_valence3(hyp_params,test_predict_data,test_true_data,name):

    cm = confusion_matrix(test_true_data, test_predict_data) 
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_valence3(hyp_params,last_features,true_data,name):
    
    target_names = ['Negative', 'Neutral', 'Positive']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10', 3), edgecolor='none', s=30)
    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 3)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
    
def save_confusion_matrix_arousal3(hyp_params,test_predict_data,test_true_data,name):

    cm = confusion_matrix(test_true_data, test_predict_data) 
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Low', 'Neutral', 'High'], 
                yticklabels=['Low', 'Neutral', 'High']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_arousal3(hyp_params,last_features,true_data,name):
    
    target_names = ['Low', 'Neutral', 'High']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10', 3), edgecolor='none', s=30)
    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 3)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)

def save_confusion_matrix_VA(hyp_params,test_predict_data,test_true_data,name):
    
    cm = confusion_matrix(test_true_data, test_predict_data)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['LVLA', 'LVHA','HVLA','HVHA'], 
                yticklabels=['LVLA', 'LVHA','HVLA','HVHA']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_VA(hyp_params,last_features,true_data,name):
    
    target_names = ['LVLA','LVHA','HVLA','HVHA']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10', 4), edgecolor='none', s=30)

    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 4)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
    
def save_confusion_matrix_wesad3(hyp_params,test_predict_data,test_true_data,name):

    cm = confusion_matrix(test_true_data, test_predict_data) 
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['BL', 'ST', 'REC'], 
                yticklabels=['BL', 'ST', 'REC']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_wesad3(hyp_params,last_features,true_data,name):
    
    target_names = ['BL', 'ST', 'REC']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10', 3), edgecolor='none', s=30)
    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 3)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
def save_confusion_matrix_wesad4(hyp_params,test_predict_data,test_true_data,name):
    
    cm = confusion_matrix(test_true_data, test_predict_data)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['BL', 'ST','REC','MED'], 
                yticklabels=['BL', 'ST','REC','MED']) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig(f"{hyp_params.confusion_matrix_path}/{name}.png",dpi=1200)

def save_tSNE_wesad4(hyp_params,last_features,true_data,name):
    
    target_names = ['BL', 'ST','REC','MED']
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(last_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_data, cmap=plt.get_cmap('tab10', 4), edgecolor='none', s=30)

    handles = []
    for i, target_name in enumerate(target_names):
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.get_cmap('tab10', 4)(i), markersize=10, linestyle='None', label=target_name)
        handles.append(handle)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fancybox=True, shadow=True, framealpha=1)
    plt.savefig(f"{hyp_params.tSNE_path}/{name}.png",dpi=1200)
        
def update_plot_td(epoch, train_loss_list,valid_loss_list,test_loss_list,train_accuracy_list,valid_accuracy_list,test_accuracy_list, 
                train_precision_list,valid_precision_list,test_precision_list,train_recall_list,valid_recall_list,test_recall_list,
                train_f1_list,valid_f1_list,test_f1_list,save_path):
    font_size = 36  
    fig, axes = plt.subplots(5, 3, figsize=(60, 45))

    # Convert tensors to numpy arrays
    train_loss = to_numpy(train_loss_list)
    valid_loss = to_numpy(valid_loss_list)
    test_loss = to_numpy(test_loss_list)
    train_accuracy = to_numpy(train_accuracy_list)
    valid_accuracy = to_numpy(valid_accuracy_list)
    test_accuracy = to_numpy(test_accuracy_list)
    train_f1 = to_numpy(train_f1_list)
    valid_f1 = to_numpy(valid_f1_list)
    test_f1 = to_numpy(test_f1_list)
    train_precision = to_numpy(train_precision_list)
    valid_precision = to_numpy(valid_precision_list)
    test_precision = to_numpy(test_precision_list)
    train_recall = to_numpy(train_recall_list)
    valid_recall = to_numpy(valid_recall_list)
    test_recall = to_numpy(test_recall_list)

    axes[0, 0].plot(range(1, epoch+1), train_loss, label="Loss", color='blue')
    axes[0, 0].set_title("train_loss",fontsize=font_size)
    axes[0, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 0].set_ylabel("Loss",fontsize=font_size)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 1].plot(range(1, epoch+1), valid_loss, label="Loss", color='green')
    axes[0, 1].set_title("valid_loss",fontsize=font_size)
    axes[0, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 1].set_ylabel("Loss",fontsize=font_size)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 2].plot(range(1, epoch+1), test_loss, label="Loss", color='red')
    axes[0, 2].set_title("test_loss",fontsize=font_size)
    axes[0, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 2].set_ylabel("Loss",fontsize=font_size)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 0].plot(range(1, epoch+1), train_accuracy, label="Acc", color='blue')
    axes[1, 0].set_title("Train Accuracy",fontsize=font_size)
    axes[1, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 0].set_ylabel("Accuracy",fontsize=font_size)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 1].plot(range(1, epoch+1), valid_accuracy, label="Acc", color='green')
    axes[1, 1].set_title("Valid Accuracy",fontsize=font_size)
    axes[1, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 1].set_ylabel("Accuracy",fontsize=font_size)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 2].plot(range(1, epoch+1), test_accuracy, label="Acc", color='red')
    axes[1, 2].set_title("Test Accuracy",fontsize=font_size)
    axes[1, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 2].set_ylabel("Accuracy",fontsize=font_size)
    axes[1, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[2, 0].plot(range(1, epoch+1), train_precision, label="Precision", color='blue')
    axes[2, 0].set_title("Train Precision",fontsize=font_size)
    axes[2, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[2, 0].set_ylabel("Precision",fontsize=font_size)
    axes[2, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[2, 1].plot(range(1, epoch+1), valid_precision, label="Precision", color='green')
    axes[2, 1].set_title("Valid Precision",fontsize=font_size)
    axes[2, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[2, 1].set_ylabel("Precision",fontsize=font_size)
    axes[2, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[2, 2].plot(range(1, epoch+1), test_precision, label="Precision", color='red')
    axes[2, 2].set_title("Test Precision",fontsize=font_size)
    axes[2, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[2, 2].set_ylabel("Precision",fontsize=font_size)
    axes[2, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[3, 0].plot(range(1, epoch+1), train_recall, label="Recall", color='blue')
    axes[3, 0].set_title("Train Recall",fontsize=font_size)
    axes[3, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[3, 0].set_ylabel("Recall",fontsize=font_size)
    axes[3, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[3, 1].plot(range(1, epoch+1), valid_recall, label="Recall", color='green')
    axes[3, 1].set_title("Valid Recall",fontsize=font_size)
    axes[3, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[3, 1].set_ylabel("Recall",fontsize=font_size)
    axes[3, 1].tick_params(axis='both', which='major', labelsize=32) 
    axes[3, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[3, 2].plot(range(1, epoch+1), test_recall, label="Recall", color='red')
    axes[3, 2].set_title("Test Recall",fontsize=font_size)
    axes[3, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[3, 2].set_ylabel("Recall",fontsize=font_size)
    axes[3, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[4, 0].plot(range(1, epoch+1), train_f1, label="F1", color='blue')
    axes[4, 0].set_title("Train F1 Score",fontsize=font_size)
    axes[4, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[4, 0].set_ylabel("F1 Score",fontsize=font_size)
    axes[4, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[4, 1].plot(range(1, epoch+1), valid_f1, label="F1", color='green')
    axes[4, 1].set_title("Valid F1 Score",fontsize=font_size)
    axes[4, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[4, 1].set_ylabel("F1 Score",fontsize=font_size)
    axes[4, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[4, 2].plot(range(1, epoch+1), test_f1, label="F1", color='red')
    axes[4, 2].set_title("Test F1 Score",fontsize=font_size)
    axes[4, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[4, 2].set_ylabel("F1 Score",fontsize=font_size)
    axes[4, 2].tick_params(axis='both', which='major', labelsize=32) 
    
    # Add legends and grid to each subplot
    for row in axes:
        for ax in row:
            ax.legend()
            ax.grid()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    
def update_plot_sd(epoch, train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list, 
                train_precision_list,test_precision_list,train_recall_list,test_recall_list,
                train_f1_list,test_f1_list,save_path):
    
    font_size = 36 
    fig, axes = plt.subplots(2, 5, figsize=(60, 30))

    # Convert tensors to numpy arrays
    train_loss = to_numpy(train_loss_list)
    test_loss = to_numpy(test_loss_list)
    train_accuracy = to_numpy(train_accuracy_list)
    test_accuracy = to_numpy(test_accuracy_list)
    train_f1 = to_numpy(train_f1_list)
    test_f1 = to_numpy(test_f1_list)
    train_precision = to_numpy(train_precision_list)
    test_precision = to_numpy(test_precision_list)
    train_recall = to_numpy(train_recall_list)
    test_recall = to_numpy(test_recall_list)

    axes[0, 0].plot(range(1, epoch+1), train_loss, label="Loss", color='blue')
    axes[0, 0].set_title("train_loss",fontsize=font_size)
    axes[0, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 0].set_ylabel("Loss",fontsize=font_size)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 1].plot(range(1, epoch+1), train_accuracy, label="Acc", color='blue')
    axes[0, 1].set_title("Train Accuracy",fontsize=font_size)
    axes[0, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 1].set_ylabel("Accuracy",fontsize=font_size)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 2].plot(range(1, epoch+1), train_precision, label="Precision", color='blue')
    axes[0, 2].set_title("Train Precision",fontsize=font_size)
    axes[0, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 2].set_ylabel("Precision",fontsize=font_size)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 3].plot(range(1, epoch+1), train_recall, label="Recall", color='blue')
    axes[0, 3].set_title("Train Recall",fontsize=font_size)
    axes[0, 3].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 3].set_ylabel("Recall",fontsize=font_size)
    axes[0, 3].tick_params(axis='both', which='major', labelsize=32) 

    axes[0, 4].plot(range(1, epoch+1), train_f1, label="F1", color='blue')
    axes[0, 4].set_title("Train F1 Score",fontsize=font_size)
    axes[0, 4].set_xlabel("Epoch",fontsize=font_size)
    axes[0, 4].set_ylabel("F1 Score",fontsize=font_size)
    axes[0, 4].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 0].plot(range(1, epoch+1), test_loss, label="Loss", color='red')
    axes[1, 0].set_title("test_loss",fontsize=font_size)
    axes[1, 0].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 0].set_ylabel("Loss",fontsize=font_size)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 1].plot(range(1, epoch+1), test_accuracy, label="Acc", color='red')
    axes[1, 1].set_title("Test Accuracy",fontsize=font_size)
    axes[1, 1].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 1].set_ylabel("Accuracy",fontsize=font_size)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 2].plot(range(1, epoch+1), test_precision, label="Precision", color='red')
    axes[1, 2].set_title("Test Precision",fontsize=font_size)
    axes[1, 2].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 2].set_ylabel("Precision",fontsize=font_size)
    axes[1, 2].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 3].plot(range(1, epoch+1), test_recall, label="Recall", color='red')
    axes[1, 3].set_title("Test Recall",fontsize=font_size)
    axes[1, 3].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 3].set_ylabel("Recall",fontsize=font_size)
    axes[1, 3].tick_params(axis='both', which='major', labelsize=32) 

    axes[1, 4].plot(range(1, epoch+1), test_f1, label="F1", color='red')
    axes[1, 4].set_title("Test F1 Score",fontsize=font_size)
    axes[1, 4].set_xlabel("Epoch",fontsize=font_size)
    axes[1, 4].set_ylabel("F1 Score",fontsize=font_size)
    axes[1, 4].tick_params(axis='both', which='major', labelsize=32) 

    # Add legends and grid to each subplot
    for row in axes:
        for ax in row:
            ax.legend()
            ax.grid()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    


