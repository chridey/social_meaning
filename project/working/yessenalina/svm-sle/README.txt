-------------------------
README File For SVM-sle
-------------------------
Yisong Yue

Version 1.00
09/08/2010

http://projects.yisongyue.com/svmsle/


-------------------------
INTRODUCTION
-------------------------

SVM-sle is an SVM method for learning models that predict document-level sentiment.  The models are structured two-level models that also solves an extraction subtask which identifies that best supporting sentences for the document-level sentiment.  See [1] for more details.  SVM-sle is built on top of the SVM-Struct package.  Please refer to README_STRUCT.txt for further information.


-------------------------
COMPILING
-------------------------

To compile, simply run 'make' in the svm-sle directory.  To compile on a Windows machine, you will likely need to use Cygwin or some other program that emulates a Linux environment.


-------------------------
INPUT DATA FORMAT
-------------------------

The input data file which SVM-sle reads is one file of documents broken down into sentences.  This data format is used for both training and classifying (for training, an additional input file is required -- see LEARNING below for more details).

Documents are broken down into contiguous groups of lines, with each sentence represented by one line.  The first line consists of the document label (1 or -1) and the number of sentences.  Each subsequent line corresponds to the feature vector describing a single sentence.  The first entry in the sentence line corresponds to that sentence's index (starting from 0).  

Features are represented sparsely.  For each document, only the non-zero feature values need to be stored in the data file.  

SENTENCE-LEVEL FEATURES:
There are two types of sentence features, polarity and subjectivity features.  Polarity features follow the form [id]:[value], where [id] is the polarity feature id.  Subjectivity features follow the form S[id]:[value], where [id] is the subjectivity feature id.  Polarity features must be listed before subjectivity features, and the ids must be sorted in ascending order.

DOCUMENT-LEVEL FEATURES:
The very last line (which starts with index value equal to the number of sentences) is a feature vector describing the entire document.  This can contain only polarity features.  If you are using the feature smoothing mode (using flag '-m 3' -- this is also the DEFAULT feature mode), then these features must also correspond to the sentence-level polarity features.  For example, if document-level feature 3 corresponds to the frequency of the word 'like' in the entire document, the sentence-level polarity feature 3 should correspond to the frequence of the word 'like' in that particular sentence.

EXAMPLE:

-1 3
0 1:1 3:0.5 5:1 S2:1 S4:1
1 2:1 3:1 S1:0.8
2 4:1 5:1 6:1 S2:1 S3:1
3 1:0.5 2:0.1 3:1.5 4:0.2

The first line correspond to [label] [#sentences], which is this case indicates that the document has a negative polarity and contains three sentences.

The next three lines each correspond to one of the sentences.  The first entry in each line is the sentence index (from 0 to 2).  Consider the first sentence (which has index 0).  This sentence has three polarity features (with feature IDs 1, 3, 5) and two subjectivity features (with feature IDs 2, 4).  

The last line (which has index 3) holds the document-level features.  Only polarity features can be used here.  If you are using the feature smoothing mode (using flag '-m 3' -- this is also the DEFAULT feature mode), then these feature ids to correspond to the sentence-level polarity  feature ids. 

Documents are separated by one empty line.  See the data_sample folder for more examples.


-------------------------
LEARNING
-------------------------

After the program is compiled, the executable to use for learning is svm_sle_learn.  Use the following usage pattern:

svm_sle_learn -c [c_value] [data_file] [latent_file] [model_file]

where c_value is the C parameter which controls the tradeoff between regularization and training loss, data_file is the data file, latent_file is the file containing the current guess of the best extracted explanations, and model_file is the file to write the trained model to. 

*****NOTE***** 
This is very important!  The c code only implements one iteration of learning.  In particular,  The c code only implements the SSVMSolve subroutine of Algorithm 2 in [1].  Each iteration of training utilizes a new latent_file.  See run_training.py for an example of how to run training for multiple iterations.


EXTRACTION SIZE MODES
You might want to specify the extraction size constraint, as described in Eq. (5) in [1].  This is controlled by the '-l n' flag.  For n between 1 and 99, this is interpreted as a percentage of the total number of sentences in the document, rounded to the nearest integer, and lower capped at 1.  The default setting is 30, which was used in the experiments in [1].  You can implement more by modifying latent_size_custom.cc.


FEATURE MODES
You might want to specify the feature mode, which controls how to combine document-level and sentence-level features. This is controlled by the '-m' flag in svm_sle_learn.  The options are described below.

0 -- flat mode.  This mode only looks at the document-level features and ignores sentence level features.  Functionally identical to a standard SVM (although the use of C values between svm-sle and svm-light are not directly equivalent).  This mode is useful for learning priors as discussed in Section 4.5.1 in [1].  When training in this mode, latent_file is ignored.

1 -- sentence-only mode.  This mode only looks at the sentence-level features, and corresponds to Eq. (3) in [1].

2 -- combines modes 0 + 1.  This mode combines the two previous modes, but does not tie the features together in any way.  Basically, each mode gets a vote on the polarity of the document.

3 -- feature smoothing mode, (DEFAULT).  This implements Section 4.5.2 in [1].  This mode is simlar to mode 2 except that the sentence-level features and document-level features are tied together as described in Section 4.5.2.


NORM MODES
You might want to specify the normalization mode for the joint sentence-level features.  This corresponds to the N(x) function in Eq. (3) in [1].  The options are described below.  You can implement your own by modifying latent_size_custom.cc.

0 -- no normalization
1 -- divide by the extraction size
2 -- divide by the sqrt of the extraction size (DEFAULT).


WINDOW MODE
The contiguous window mode can be turned on by specifying the '-i' flag.  In contiguous window mode, the extracted explanations are constratined to be consecutive sentences.  This was found to actually degrade performance in preliminary experiments not reported in [1].  But the option is available.



-------------------------
CLASSIFYING
-------------------------

After the program is compiled and a model is learned, the executable to use for classifying is svm_sle_classify.  Use the following usage pattern:

svm_sle_classify [data_file] [model_file] [output_file]

where data_file is the data file holding instances to be classified, model_file is the model to use for classification, and output_file is the file to write the classification output to.  The output format is [label] [extracted sentence indexes].

You can also specify the '-a' option to output verbose.  This will output four sentence-level scores on a line for each sentence,

[polar] [subj] [joint pos] [joint neg]

where polar is the polarity score of the sentence, subj is the subjectivity score of the sentence, joint pos is the joint positive score (subj + polar), and joint neg is the joint negative score (subj - polar).  

INFERRING LATENT EXTRACTIONS
You can use svm_sle_classify to infer the best extractions by using the '-l' option.  This corresponds to Line 7 of Algorithm 2 in [1].  This is used when training for multiple iterations.  See run_training.py for an example.



-------------------------
REFERENCES
-------------------------

[1] "Multi-level Structured Models for Document-level Sentiment Classification",
     by A. Yessenalina, Y. Yue, and C. Cardie,
     In Proceedings of EMNLP 2010. 

[2] "Learning Structural SVMs with Latent Variables",
    by C.-N. Yu and Thorsten Joachims,
    In Proceedings of ICML, 2009.

[3] "Cutting Plane Training of Structural SVMs",
    by T. Joachims, T. Finley, and C.-N. Yu,
    Machine Learning, 77(1):27-59, 2009.
