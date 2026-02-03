import numpy as np
import pickle
import scipy.io
np.random.seed(2024) 

# m1, m2, m3, m4 represent the four modalities respectively. 
# This code is for the DEAP dataset. 
# For DREAMER and WESAD datasets, only minor adjustments are needed: 
# adjust the values of modalities, channels, 
# and sampling rates at the corresponding positions according to the description in the paper.

# Data preprocessing function
def data_preprocess(mat_data):
    #  Data preprocessing function: baseline correction for DEAP dataset
	#  Differences for DREAMER/WESAD datasets:
	#  1. DREAMER: Uses the last 3 seconds of the baseline period  for baseline correction.
	#  2. WESAD: No baseline correction.
	reshaped_data = mat_data.reshape(63, 128, 40)
	baseline_mean = np.mean(reshaped_data[:3], axis=0)
	corrected_data = reshaped_data[3:] - baseline_mean
	corrected_data = corrected_data.reshape(-1, 40)

	return corrected_data  

def data(mat_data,mat_label):
    # This is processing the DEAP dataset.
    # Differences for DREAMER/WESAD datasets:
	# 1. DREAMER: 
	# - Modalities: EEG (14 channels), ECG (2 channels)
	# - Labels: Valence/Arousal 1-5 scale, threshold-based partitioning
	# - Sampling rates: 128Hz (EEG), 256Hz (ECG)
	# - Tasks: 3-class (ratings of 1-2 as negative/low, 3 as neutral, and 4-5 as positive/high)
		
	# 2. WESAD:
	# - Modalities: ECG, EMG, EDA
	# - Labels: Discrete states (baseline, stress, recreation, meditation)
	# - Sampling rates: 700Hz (chest)
	# - Tasks include:(i) 3-class (baseline, stress, recreation) and (ii) 4-class (baseline, stress, recreation, meditation).
 
    
    # Initialize lists for each modality
	m1_data = []
	m2_data = []
	m3_data = []
	m4_data = []
 
 	# Temporary lists for unsegmented data
	m1_data1 = []
	m2_data1 = []
	m3_data1 = []
	m4_data1 = []
	label_data = []

	# Apply baseline correction
	mat_data = data_preprocess(mat_data)
	for k in range(mat_data.shape[0]): 
		m1_data1.append(mat_data[k][:32]) # EEG
		m2_data1.append(mat_data[k][32:34]) # EOG
		m3_data1.append(mat_data[k][34:36]) # EMG
		m4_data1.append(mat_data[k][36]) # GSR
    # Segment data into 1-second windows
	for i in range(0,len(m1_data1),128): 
		m1_data.append(m1_data1[i:i+128])
		m2_data.append(m2_data1[i:i+128])
		m3_data.append(m3_data1[i:i+128])
		m4_data.append(m4_data1[i:i+128])
  
 		# labels : mat_label1[0]->valence, mat_label1[1]->arousal 
		if 1<=round(mat_label[1])<=3: 
			label_data.append(0)
		elif 4<=round(mat_label[1])<=6:
			label_data.append(1)
		elif 7<=round(mat_label[1])<=9:
			label_data.append(2)
	data_len = len(m1_data) 
	return m1_data,m2_data,m3_data,m4_data,label_data,data_len

# Create and save train/validation/test splits as pickle files.
def pkl_make(m1,m2,m3,m4,label,train_id,val_id,test_id,pkl,epoch):
    #The number of modes, channels, and sampling rates at the corresponding positions of DREAMER and WESAD need to change. 
	print('data over'+ str(epoch))

	m1_train = np.array(m1)[train_id].reshape(train_id.shape[0],32,128)
	m1_val = np.array(m1)[val_id].reshape(val_id.shape[0],32,128)
	m1_test = np.array(m1)[test_id].reshape(test_id.shape[0],32,128)

	m2_train = np.array(m2)[train_id].reshape(train_id.shape[0],2,128)
	m2_val = np.array(m2)[val_id].reshape(val_id.shape[0],2,128)
	m2_test = np.array(m2)[test_id].reshape(test_id.shape[0],2,128)

	m3_train = np.array(m3)[train_id].reshape(train_id.shape[0],2,128)
	m3_val = np.array(m3)[val_id].reshape(val_id.shape[0],2,128)
	m3_test = np.array(m3)[test_id].reshape(test_id.shape[0],2,128)

	m4_train =  np.array(m4)[train_id].reshape(train_id.shape[0],1,128)
	m4_val = np.array(m4)[val_id].reshape(val_id.shape[0],1,128)
	m4_test = np.array(m4)[test_id].reshape(test_id.shape[0],1,128)

	id_train = np.arange(train_id.shape[0]).reshape(train_id.shape[0],1,1)
	id_val = np.arange(val_id.shape[0]).reshape(val_id.shape[0],1,1)
	id_test = np.arange(test_id.shape[0]).reshape(test_id.shape[0],1,1)
	
	label_train = np.array(label)[train_id].reshape(train_id.shape[0],1,1)
	label_val = np.array(label)[val_id].reshape(val_id.shape[0],1,1)
	label_test = np.array(label)[test_id].reshape(test_id.shape[0],1,1)

	# Create nested dictionary structure for pickle file
	pkl1 = {}
	train = {}
	test = {}
	valid ={}

	train['id'] = id_train
	train['modality1'] = m1_train
	train['modality2'] = m2_train
	train['modality3'] = m3_train
	train['modality4'] = m4_train
	train['label'] = label_train

	valid['id'] = id_val
	valid['modality1'] = m1_val
	valid['modality2'] = m2_val
	valid['modality3'] = m3_val
	valid['modality4'] = m4_val
	valid['label'] = label_val

	test['id'] = id_test
	test['modality1'] = m1_test
	test['modality2'] = m2_test
	test['modality3'] = m3_test
	test['modality4'] = m4_test
	test['label'] = label_test

	pkl1['train'] = train
	pkl1['valid'] = valid
	pkl1['test'] = test
 
	# Save to pickle file
	pickle.dump(pkl1,pkl)
	print('done'+ str(epoch))
	return

# Perform total dataset 10-fold cross-validation split for DEAP dataset.
def DEAP(array,lenth,m_1,m_2,m_3,m_4,label1):
    
	for i in range(10):
		train1 = []
		val_start = int(i*lenth/10)
		val_end = test_start = int((i+1)*lenth/10)
		test_end = int((i+2)*lenth/10)
		final_test = int(0.1*lenth)
		if i < 9:
			val = array[val_start:val_end]
			test = array[test_start:test_end]
		else:
			val = array[val_start:val_end]
			test = array[:final_test]
		for k in array: 
			if k not in np.append(val,test):
				train1.append(k)
		train = np.array(train1)
		pkl_data = open('/ULER/data_pre/deap/valence3/data_'+str(i)+'.pkl','wb')
		pkl_make(m_1,m_2,m_3,m_4,label1,train,val,test,pkl_data,i)
		
	return 

# Main execution block
if __name__ == '__main__':

	# Load list of dataset file paths
	txt = open('/ULER/data_pre/DEAP_list.txt','r').readlines()
	
	# Initialize containers for all subjects' data
	m_1 = []
	m_2 = []
	m_3 = []
	m_4 = []
	label = []
	# Process each subject file
	for i,line in enumerate(txt):
		mat_path = line.rstrip('\n')
		mat_cont = scipy.io.loadmat(mat_path)
		mat_data = np.transpose(mat_cont['data'], (0, 2, 1)) 
		mat_label = mat_cont['labels'] 
		for j in range(mat_data.shape[0]):
			mat_data = mat_data[j] 
			mat_label = mat_label[j] 
			m1_data,m2_data,m3_data,m4_data,label_data,data_len = data(mat_data,mat_label)
			m_1.extend(m1_data)
			m_2.extend(m2_data)
			m_3.extend(m3_data)
			m_4.extend(m4_data)
			label.extend(label_data)
	indices = np.arange(len(m_1)) 
	np.random.shuffle(indices)
	DEAP(indices,indices.shape[0],m_1,m_2,m_3,m_4,label)
