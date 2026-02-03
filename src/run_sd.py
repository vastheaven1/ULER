import torch
from torch import nn
from src.model import ULER_Model
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
import datetime  
from src.dataloader_sd import data_loader
from src.utils import update_plot_sd,save_confusion_matrix_valence2,save_tSNE_valence2

np.random.seed(2024) 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

# Main training class for ULER model
class ModelTraining:
    def __init__(self, args):
        self.args = args

    def train_begin(self):
        # Initialize model, optimizer, loss function and scheduler
        hyp_params = self.args
        model = ULER_Model(hyp_params).to(device) 
        optimizer = optim.Adam(model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.patience, factor=hyp_params.factor) 
        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler
                    }
        # Load data for data
        train_loader, test_loader=data_loader(self.args)
        # Start training process
        self.train_model(settings, hyp_params, train_loader, test_loader)

    def train_model(self,settings, hyp_params, train_loader, test_loader):
        model = settings['model']
        optimizer = settings['optimizer']
        criterion = settings['criterion']
        scheduler = settings['scheduler']
        model.to(device)

        # Lists to track training metrics
        train_loss_list = []
        test_loss_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        train_f1_list = []
        test_f1_list = []
        train_precision_list = []
        test_precision_list = []
        train_recall_list = []
        test_recall_list = []
         # Track best test accuracy
        best_test_accuracy = 0.00
        name = hyp_params.model + '_' + hyp_params.data_index + '_' + hyp_params.exp_number

        # Open log file for recording training progress
        with open(hyp_params.logs_path+'/'+hyp_params.dataset+'_'+hyp_params.data_index+hyp_params.exp_number+'.txt', 'a') as log_file:
            for epoch in range(1, hyp_params.num_epochs+1): 
                train_accuracy,train_precision,train_recall,train_f1,train_loss,train_epoch_time,train_predict_data,train_true_data = self.train(hyp_params,model, optimizer, criterion,train_loader)
                test_accuracy,test_precision,test_recall,test_f1,test_loss,test_epoch_time,test_predict_data,test_true_data,test_last_features = self.evaluate(hyp_params,model, criterion, test_loader, test=True)
              
                scheduler.step(test_loss)    
                # Store metrics for this epoch
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                train_f1_list.append(train_f1)
                test_f1_list.append(test_f1)
                train_precision_list.append(train_precision)
                test_precision_list.append(test_precision)
                train_recall_list.append(train_recall)
                test_recall_list.append(test_recall)

                # Update training plots
                update_plot_sd(epoch, train_loss_list,test_loss_list,train_accuracy_list,test_accuracy_list, 
                train_precision_list,test_precision_list,train_recall_list,test_recall_list,
                train_f1_list,test_f1_list,hyp_params.pictures_path+'/'+'_'+hyp_params.dataset+'_'+hyp_params.data_index+hyp_params.exp_number)

                # Log training progress
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                train_log_message = f'{current_time} - Epoch [{epoch}/{hyp_params.num_epochs}], Train_loss: {train_loss:.4f}, '\
                f'Train_ac: {train_accuracy:.4f}, Train_pr: {train_precision:.4f},Train_re: {train_recall:.4f},Train_f1: {train_f1:.4f}, Train_time: {train_epoch_time:.4f}s'
                test_log_message = f'{current_time} - Epoch [{epoch}/{hyp_params.num_epochs}], Test_loss: {test_loss:.4f}, '\
                f'Test_ac: {test_accuracy:.4f}, Test_pr: {test_precision:.4f},Test_re: {test_recall:.4f},Test_f1: {test_f1:.4f}, Test_time: {test_epoch_time:.4f}s'
                print(train_log_message)
                print(test_log_message)
                # Write to log file
                log_file.write(train_log_message + '\n')
                log_file.write(test_log_message + '\n')
                n_parameters = sum(p.numel() for p in model.parameters())
                print('n_parameters:',n_parameters)
              
                if test_accuracy > best_test_accuracy and test_accuracy > hyp_params.accuracy_threshold: 
                    
                    print(f"{hyp_params.models_path}/{hyp_params.model+hyp_params.data_index}.pt!")
                    save_model(hyp_params, model, name) 
                    save_best_result_sd(hyp_params,train_log_message,test_log_message,name) 
                    save_label(hyp_params,test_predict_data,test_true_data,name)
                    save_last_features(hyp_params,test_last_features,name) 
                    best_test_accuracy = test_accuracy
            
            # After training, generate confusion matrix and t-SNE visualization
            predict_data,true_data = load_label(hyp_params,name)    
            # For different tasks, different save functions should be selected.
            save_confusion_matrix_valence2(hyp_params,predict_data,true_data,name)
            last_features = load_last_features(hyp_params,name)
            save_tSNE_valence2(hyp_params,last_features,true_data,name)

    def train(self,hyp_params,model, optimizer, criterion,train_loader):
        """Training function for one epoch"""
        model.train() # Set model to training mode
        epoch_loss = 0.0
        epoch_size = 0
        # Lists to collect predictions and true labels
        predict_data_list = []
        true_data_list = []
        start_time = time.time()
        # Iterate over training batches
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            # This code is for the DEAP dataset. 
            # For the DREAMER dataset, only m1 and m2 should be used, 
            # and for the WESAD dataset, only m1, m2, and m3 should be used.
            sample_ind, m1,m2,m3,m4 = batch_X
            eval_attr = batch_Y.squeeze().long()
            model.zero_grad()
            if hyp_params.use_cuda:
                    m1,m2,m3,m4,eval_attr = m1.to(device),m2.to(device),m3.to(device),m4.to(device),eval_attr.to(device)
            batch_size = m1.size(0) # 1024
            preds, last_features = model(m1,m2,m3,m4)   
            _, predicted = torch.max(preds, 1)
            predict_data_list.append(predicted)
            true_data_list.append(eval_attr)
            batch_loss = criterion(preds, eval_attr) 
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip) 
            optimizer.step()
            epoch_size += batch_size 
            epoch_loss += batch_loss.item() * batch_size 
        # Concatenate predictions and true labels
        predict_data = torch.cat(predict_data_list, dim=0).cpu().numpy()
        true_data = torch.cat(true_data_list, dim=0).cpu().numpy()
        # Calculate metrics
        accuracy = accuracy_score(true_data, predict_data)
        precision = precision_score(true_data, predict_data, average='weighted')
        recall = recall_score(true_data, predict_data, average='weighted')
        f1 = f1_score(true_data, predict_data, average='weighted')
        avg_loss = epoch_loss / epoch_size 
        epoch_time = time.time() - start_time 
        
        return accuracy,precision,recall,f1,avg_loss,epoch_time,predict_data,true_data
       
    def evaluate(self,hyp_params,model, criterion, test_loader, test=False):
        """Evaluation function"""
        model.eval()
        loader = test_loader 
        epoch_loss = 0.0
        epoch_size = 0
        # Lists to collect predictions, true labels and features
        predict_data_list = []
        true_data_list = []
        last_features_list = []
        start_time = time.time()
        with torch.no_grad():
            # Iterate over evaluation batches
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                # This code is for the DEAP dataset. 
                # For the DREAMER dataset, only m1 and m2 should be used, 
                # and for the WESAD dataset, only m1, m2, and m3 should be used.
                sample_ind,m1,m2,m3,m4 = batch_X
                eval_attr = batch_Y.squeeze().long()
                if hyp_params.use_cuda:
                        m1,m2,m3,m4,eval_attr = m1.to(device),m2.to(device),m3.to(device),m4.to(device),eval_attr.to(device)  
                batch_size = m1.size(0)
                preds, last_features = model(m1,m2,m3,m4)
                _, predicted = torch.max(preds, 1)
                predict_data_list.append(predicted)
                true_data_list.append(eval_attr)
                last_features_list.append(last_features)
                epoch_size += batch_size 
                epoch_loss += criterion(preds, eval_attr).item() * batch_size
        predict_data = torch.cat(predict_data_list, dim=0).cpu().numpy()
        true_data = torch.cat(true_data_list, dim=0).cpu().numpy()
        last_features_data = torch.cat(last_features_list, dim=0).cpu().numpy()
        # Calculate metrics
        accuracy = accuracy_score(true_data, predict_data)
        precision = precision_score(true_data, predict_data, average='weighted')
        recall = recall_score(true_data, predict_data, average='weighted')
        f1 = f1_score(true_data, predict_data, average='weighted')
        avg_loss = epoch_loss / epoch_size
        epoch_time = time.time() - start_time 
        
        return accuracy,precision,recall,f1,avg_loss,epoch_time,predict_data,true_data,last_features_data
    
