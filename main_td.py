import torch
from src.utils import *
# from src import train
from src.run_td import ModelTraining
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") 

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# Create hyperparameter configuration for the experiment
def create_hyp_params(data_index):
  
    # Set base paths for data and results.
    # For other Total Dataset tasks, the names can be modified here to the corresponding ones, 
    # such as the arousal dimension of the DEAP dataset, or the relevant tasks for the DREAMER and WESAD datasets.
    base_path = '/ULER/data_pre/deap/valence3'
    output_path = '/ULER/results/DEAP/valence3'
    exp_number = f'DEAP_valence3' 
    
    # Create all required output directories
    ensure_directory_exists(base_path)
    ensure_directory_exists(output_path)
    ensure_directory_exists(f'{output_path}/output_logs')
    ensure_directory_exists(f'{output_path}/output_models')
    ensure_directory_exists(f'{output_path}/output_pictures')
    ensure_directory_exists(f'{output_path}/out_labels')
    ensure_directory_exists(f'{output_path}/output_confusion_matrix')
    ensure_directory_exists(f'{output_path}/output_best_result')
    ensure_directory_exists(f'{output_path}/output_last_features')
    ensure_directory_exists(f'{output_path}/output_tSNE')
    
    # Create hyperparameter dictionary
    hyp_params = {
        
        'model': 'ULER',
        # Data configuration
        'dataset': 'DEAP',
        'data_path': base_path,
        'logs_path': f'{output_path}/output_logs',
        'models_path': f'{output_path}/output_models',
        'pictures_path': f'{output_path}/output_pictures',
        'labels_path': f'{output_path}/out_labels',
        'confusion_matrix_path': f'{output_path}/output_confusion_matrix',
        'best_result_path': f'{output_path}/output_best_result',
        'last_features_path': f'{output_path}/output_last_features',
        'tSNE_path': f'{output_path}/output_tSNE',
        'exp_number': exp_number,
        'accuracy_threshold': 0.7,
        'data_index': data_index,
        
        # Dropout configuration
        'attn_dropout': 0.1,
        'relu_dropout': 0.1,
        'embed_dropout': 0.1,
        'res_dropout': 0.1,
        'out_dropout': 0.1,
        
        # Model architecture configuration
        'layers': 2,
        'num_heads': 3,
        'attn_mask': False,  
        
        # Training configuration
        'batch_size': 1024,
        'num_epochs': 100,
        
        # Other settings
        'seed': 2024,
        'no_cuda': False,
        'use_cuda': use_cuda,
        'device': device,
        'lr': 5e-3,
        'weight_decay': 1e-5,
        'factor': 0.2,
        'patience': 5,  
        'category': 3,
        'clip': 0.8,
        'criterion': 'CrossEntropyLoss'
    }
    
    return hyp_params

# Main training function
def main(data_index):
    
    hyp_params_dict = create_hyp_params(data_index)
    torch.manual_seed(hyp_params_dict['seed'])
    torch.cuda.manual_seed_all(hyp_params_dict['seed'])
    np.random.seed(hyp_params_dict['seed'])
    class SimpleObject:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    hyp_params = SimpleObject(**hyp_params_dict)
    
    # Print configuration information
    print(f"\n=== Start Training ===")
    print(f"Data index: {data_index}")
    
    train = ModelTraining(hyp_params)
    train.train_begin()

if __name__ == '__main__':
    
    # Total dataset
    start_idx = 0  
    end_idx = 10   
    for i in range(start_idx, end_idx):
        data_index = f"data_{i}"
        main(data_index)



        
