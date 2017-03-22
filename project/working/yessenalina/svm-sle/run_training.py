import sys, os

#USAGE:  python run_training.py NumIter C train init model test

# note that the c code only implements 1 iteration of the training process
# (Algorithm 2 in the paper).

# this python script is a wrapper that runs the c code (svm_sle_learn/classify)
# for multiple iterations

# this script will output  multiple intermediate files so you can do model 
# selection post-processing


# Explanation of input parameters:

# NumIter is the number of training iterations (e.g., 10)
# C is the SVM C parameter
# train is the path to the training file
# init is the path to the latent variable initialization file
# is the file-prefix of the model files to be outputted 
# test is the path to the test file

# Example: 
# python run_training.py 10 1000 data_sample/train.txt data_sample/train.init.last25 model_C1000  data_sample/test.txt

# Note: In general you should also compare against a validation set when doing model selection

n = int(sys.argv[1])
c = sys.argv[2]
m = '3' # feature smoothing -- see Section 4.5 in paper
norm = '2' # sqrt norm -- see Eq. (3) and Section 4.3 in paper
size = '25' # extraction size -- see Eq. (5) in paper

train_file = sys.argv[3]
init_file = sys.argv[4]
model_file = sys.argv[5]
test_file = sys.argv[6]
subdir = model_file + '_temp'

# creating directory for intermediate outputs
os.system('mkdir ' + subdir)

# training first iteration using initialization file 
os.system('./svm_sle_learn -c ' + c + ' -m ' + m + ' -n ' + norm + ' -l ' + size + ' ' + train_file + ' ' + init_file + ' ' + subdir + '/model_0')

# classification performance of model from first iteration
os.system('./svm_sle_classify ' + train_file + ' ' + subdir + '/model_0 ' + subdir + '/pred_0 |tee ' + subdir + '/pred_dump_0')
os.system('./svm_sle_classify ' + test_file + ' ' + subdir + '/model_0 ' + subdir + '/test_pred_0 |tee '+ subdir + '/test_pred_dump_0')

# iteration for n iterations
for nn in xrange(1,n):
    # generating a new initilization file using model learned from previous iteration
    # see Section 4.2 in paper and Algorithm 2 Line 7
    os.system('./svm_sle_classify -l ' + train_file + ' ' + subdir + '/model_' + str(nn-1) + ' ' + subdir + '/latent_' + str(nn))

    # training new iteration
    os.system('./svm_sle_learn -c ' + c + ' -m ' + m + ' -n ' + norm + ' -l ' + size + ' ' + train_file + ' ' + subdir + '/latent_' + str(nn)+ ' ' + subdir + '/model_' + str(nn))

    # classification performance of current iteration
    os.system('./svm_sle_classify ' + train_file + ' ' + subdir + '/model_' + str(nn) + ' ' + subdir + '/pred_' + str(nn) + ' |tee ' + subdir + '/pred_dump_' + str(nn) )
    os.system('./svm_sle_classify ' + test_file + ' ' + subdir + '/model_' + str(nn) + ' ' + subdir + '/test_pred_'+str(nn) + ' |tee ' + subdir + '/test_pred_dump_' + str(nn))

