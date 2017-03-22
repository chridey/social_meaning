/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
#include "helper.h"

void        init_latent(SAMPLE *sample, STRUCTMODEL *sm, char *file)
{
    init_latent_helper(sample, sm, file);
}

void        svm_struct_learn_api_init(int argc, char* argv[])
{
    /* Called in learning part before anything else is done to allow
       any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
    /* Called in learning part at the very end to allow any clean-up
       that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
    /* Called in prediction part before anything else is done to allow
       any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
    /* Called in prediction part at the very end to allow any clean-up
       that might be necessary. */
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
    /* Reads struct examples and returns them in sample. The number of
       examples must be written into sample.n */

    return read_struct_examples_helper(file, sparm);
}

void        init_struct_model(SAMPLE *sample, STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
        KERNEL_PARM *kparm)
{
    /* Initialize structmodel sm. The weight vector w does not need to be
       initialized, but you need to provide the maximum size of the
       feature space in sizePsi. This is the maximum number of different
       weights that can be learned. Later, the weight vector w will
       contain the learned weights for the model. */

    //this logic runs during training, but not classifying
    if(sparm->model_loaded == 0)
    {
        sm->feature_mode = sparm->feature_mode;
        sm->latent_size_mode = sparm->latent_size_mode;
        sm->norm_mode = sparm->norm_mode;
        sm->sizeDocPsi = sparm->max_doc_feature_key;
        sm->sizeLatentPsi = sparm->max_feature_key;
        sm->sizeLatentPsiSubj = sparm->max_sentence_feature_key;
        sm->sizePsi = sm->sizeDocPsi + sm->sizeLatentPsi + sm->sizeLatentPsiSubj;
        sm->window_mode = sparm->window_mode;
        init_latent(sample, sm, sparm->latent_init_file);
    }
    
    //offsets the document-level feature ids
    process_document_features(sample, sm->sizeLatentPsi + sm->sizeLatentPsiSubj);

    //in MODE_FEAT_SMOOTH, we are tying together the document-level and sentence-level polarity features
    //as described in Section 4.5.2 in paper.  
    //this is implemented by duplicating the sentence level features as document-level features
    if(sm->feature_mode == MODE_FEAT_SMOOTH)
    {
        process_sentence_features(sample,sm->sizeLatentPsi + sm->sizeLatentPsiSubj);
        if(sm->sizeDocPsi < sm->sizeLatentPsi)
        {
            sm->sizeDocPsi = sm->sizeLatentPsi;
            sm->sizePsi = sm->sizeDocPsi + sm->sizeLatentPsi + sm->sizeLatentPsiSubj;
        }
    }

    // offset the subjectivity features
    process_sentence_features2(sample,sm->sizeLatentPsi);

    printf("sizePsi: %ld\n", sm->sizePsi);
    printf("sizeLatentPsi: %ld\n", sm->sizeLatentPsi);
    printf("sizeLatentPsiSubj: %ld\n", sm->sizeLatentPsiSubj);
    printf("sizeDocPsi: %ld\n", sm->sizeDocPsi);
    printf("feature mode: %d\n", sm->feature_mode);
    printf("extraction size mode: %d\n", sm->latent_size_mode);
    printf("extraction norm mode: %d\n", sm->norm_mode);
    if(sm->window_mode == 1)
        printf("window mode\n");
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    /* Initializes the optimization problem. Typically, you do not need
       to change this function, since you want to start with an empty
       set of constraints. However, if for example you have constraints
       that certain weights need to be positive, you might put that in
       here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
       is an array of feature vectors, rhs is an array of doubles. m is
       the number of constraints. The function returns the initial
       set of constraints. */

    return init_struct_constraints_helper(sample, sm, sparm);
}

LABEL       infer_hidden_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    return infer_hidden_variables_helper(x,y,sm,sparm);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    /* Finds the label yhat for pattern x that scores the highest
       according to the linear evaluation function in sm, especially the
       weights sm.w. The returned label is taken as the prediction of sm
       for the pattern x. The weights correspond to the features defined
       by psi() and range from index 1 to index sm->sizePsi. If the
       function cannot find a label, it shall return an empty label as
       recognized by the function empty_label(y). */

    return classify_struct_example_helper(x,sm,sparm);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
        STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    /* Finds the label ybar for pattern x that that is responsible for
       the most violated constraint for the slack rescaling
       formulation. For linear slack variables, this is that label ybar
       that maximizes

       argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) 

       Note that ybar may be equal to y (i.e. the max is 0), which is
       different from the algorithms described in
       [Tschantaridis/05]. Note that this argmax has to take into
       account the scoring function in sm, especially the weights sm.w,
       as well as the loss function, and whether linear or quadratic
       slacks are used. The weights in sm.w correspond to the features
       defined by psi() and range from index 1 to index
       sm->sizePsi. Most simple is the case of the zero/one loss
       function. For the zero/one loss, this function should return the
       highest scoring label ybar (which may be equal to the correct
       label y), or the second highest scoring label ybar, if
       Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
       shall return an empty label as recognized by the function
       empty_label(y). */
    LABEL ybar;

    /* insert your code for computing the label ybar here */

    return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
        STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    /* Finds the label ybar for pattern x that that is responsible for
       the most violated constraint for the margin rescaling
       formulation. For linear slack variables, this is that label ybar
       that maximizes

       argmax_{ybar} loss(y,ybar)+psi(x,ybar)

       Note that ybar may be equal to y (i.e. the max is 0), which is
       different from the algorithms described in
       [Tschantaridis/05]. Note that this argmax has to take into
       account the scoring function in sm, especially the weights sm.w,
       as well as the loss function, and whether linear or quadratic
       slacks are used. The weights in sm.w correspond to the features
       defined by psi() and range from index 1 to index
       sm->sizePsi. Most simple is the case of the zero/one loss
       function. For the zero/one loss, this function should return the
       highest scoring label ybar (which may be equal to the correct
       label y), or the second highest scoring label ybar, if
       Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
       shall return an empty label as recognized by the function
       empty_label(y). */
    return find_most_violated_constraint_marginrescaling_helper(x,y,sm,sparm);
}

int         empty_label(LABEL y)
{
    /* Returns true, if y is an empty label. An empty label might be
       returned by find_most_violated_constraint_???(x, y, sm) if there
       is no incorrect label that can be found for x, or if it is unable
       to label x at all */
    return(0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
        STRUCT_LEARN_PARM *sparm)
{
    /* Returns a feature vector describing the match between pattern x
       and label y. The feature vector is returned as a list of
       SVECTOR's. Each SVECTOR is in a sparse representation of pairs
       <featurenumber:featurevalue>, where the last pair has
       featurenumber 0 as a terminator. Featurenumbers start with 1 and
       end with sizePsi. Featuresnumbers that are not specified default
       to value 0. As mentioned before, psi() actually returns a list of
       SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
       specifies the next element in the list, terminated by a NULL
       pointer. The list can be though of as a linear combination of
       vectors, where each vector is weighted by its 'factor'. This
       linear combination of feature vectors is multiplied with the
       learned (kernelized) weight vector to score label y for pattern
       x. Without kernels, there will be one weight in sm.w for each
       feature. Note that psi has to match
       find_most_violated_constraint_???(x, y, sm) and vice versa. In
       particular, find_most_violated_constraint_???(x, y, sm) finds
       that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
       inner vector product) and the appropriate function of the
       loss + margin/slack rescaling method. See that paper for details. */
    return psi_helper(x,y,sm,sparm);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
    /* loss for correct label y and predicted label ybar. The loss for
       y==ybar has to be zero. sparm->loss_function is set with the -l option. */
    return (double)(y.sentiment * ybar.sentiment <= 0);
}

int         finalize_iteration(double ceps, int cached_constraint,
        SAMPLE sample, STRUCTMODEL *sm,
        CONSTSET cset, double *alpha, 
        STRUCT_LEARN_PARM *sparm)
{
    /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
    return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
        CONSTSET cset, double *alpha, 
        STRUCT_LEARN_PARM *sparm)
{
    /* This function is called after training and allows final touches to
       the model sm. But primarly it allows computing and printing any
       kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
        STRUCT_LEARN_PARM *sparm, 
        STRUCT_TEST_STATS *teststats)
{
    /* This function is called after making all test predictions in
       svm_struct_classify and allows computing and printing any kind of
       evaluation (e.g. precision/recall) you might want. You can use
       the function eval_prediction to accumulate the necessary
       statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
        STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
        STRUCT_TEST_STATS *teststats)
{
    /* This function allows you to accumlate statistic for how well the
       predicition matches the labeled example. It is called from
       svm_struct_classify. See also the function
       print_struct_testing_stats. */
    if(exnum == 0) { /* this is the first time the function is
                        called. So initialize the teststats */
    }
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
        STRUCT_LEARN_PARM *sparm)
{
    FILE *modelfl;
    int i;

    if((modelfl = fopen(file, "w")) == NULL)
    { perror (file); exit (1); }

    fprintf(modelfl, "# sizePsi: %ld\n", sm->sizePsi);
    fprintf(modelfl, "# sizeLatentPsi:%ld\n", sm->sizeLatentPsi);
    fprintf(modelfl, "# sizeLatentPsiSubj:%ld\n", sm->sizeLatentPsiSubj);
    fprintf(modelfl, "# sizeDocPsi:%ld\n", sm->sizeDocPsi);
    fprintf(modelfl, "# feature_mode:%d\n", sm->feature_mode);
    fprintf(modelfl, "# latent_size_mode:%d\n", sm->latent_size_mode);
    fprintf(modelfl, "# norm_mode:%d\n", sm->norm_mode);
    fprintf(modelfl, "# window_mode:%d\n", sm->window_mode);

    for(i=0; i<=sm->sizePsi; i++)
        fprintf(modelfl, "%d:%lf\n", i, sm->w[i]);

    fclose(modelfl);

}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
    /* Reads structural model sm from file file. This function is used
       only in the prediction module, not in the learning module. */
    STRUCTMODEL sm;
    sm.svm_model = NULL;

    FILE *modelfl;
    long sizePsi, sizeDocPsi, sizeLatentPsi, sizeLatentPsiSubj;
    int fnum;
    int i, feature_mode, latent_size_mode, norm_mode, window_mode;
    double fweight;

    modelfl = fopen(file,"r");
    if (modelfl==NULL) {
        printf("Cannot open model file %s for input!", file);
        exit(1);
    }

    sparm->model_loaded = 1;

    if (fscanf(modelfl, "# sizePsi:%ld\n", &sizePsi)!=1) {
        printf("Incorrect model file format for %s! No sizePsi!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# sizeLatentPsi:%ld\n", &sizeLatentPsi)!=1) {
        printf("Incorrect model file format for %s! No sizeLatentPsi!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# sizeLatentPsiSubj:%ld\n", &sizeLatentPsiSubj)!=1) {
        printf("Incorrect model file format for %s! No sizeLatentPsiSubj!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# sizeDocPsi:%ld\n", &sizeDocPsi)!=1) {
        printf("Incorrect model file format for %s! No sizeDocPsi!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# feature_mode:%d\n", &feature_mode)!=1) {
        printf("Incorrect model file format for %s! No feature_mode!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# latent_size_mode:%d\n", &latent_size_mode)!=1) {
        printf("Incorrect model file format for %s! No latent_size_mode!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# norm_mode:%d\n", &norm_mode)!=1) {
        printf("Incorrect model file format for %s! No norm_mode!\n", file);
        fflush(stdout); 
    }

    if (fscanf(modelfl, "# window_mode:%d\n", &window_mode)!=1) {
        printf("Incorrect model file format for %s! No window_mode!\n", file);
        fflush(stdout); 
    }

    sm.sizePsi = sizePsi;
    sm.sizeLatentPsi = sizeLatentPsi;
    sm.sizeLatentPsiSubj = sizeLatentPsiSubj;
    sm.sizeDocPsi = sizeDocPsi;
    sm.feature_mode = feature_mode;
    sm.latent_size_mode = latent_size_mode;
    sm.norm_mode = norm_mode;
    sm.window_mode = window_mode;
    sparm->max_feature_key = sm.sizeLatentPsi;
    sparm->max_sentence_feature_key = sm.sizeLatentPsiSubj;
    sparm->max_doc_feature_key = sm.sizeDocPsi;

    sm.w = (double*)malloc(sizeof(double)*(sizePsi+1));
    for (i=0;i<sizePsi+1;i++) {
        sm.w[i] = 0.0;
    }

    while (!feof(modelfl)) {
        fscanf(modelfl, "%d:%lf", &fnum, &fweight);
        sm.w[fnum] = fweight;
    }
    fclose(modelfl); 

    sparm->model_loaded = 1;

    return(sm);
}

void        write_latent(FILE *fp, LABEL y)
{
    int i;
    fprintf(fp, "%d ", y.n);
    for(i=0; i<y.n; i++)
        fprintf(fp, "%d ", y.sentence_index[i]);
    fprintf(fp,"\n");
}

void        write_label_verbose(FILE *fp, LABEL y)
{
    int i;
    double s, s2;
    fprintf(fp, "%d ", y.sentiment);
    for(i=0; i<y.n; i++)
        fprintf(fp, "%d ", y.sentence_index[i]);
    fprintf(fp,"\n");
    fprintf(fp, "SENTENCE POS SCORE: %f\n", y.pos_score);
    fprintf(fp, "SENTENCE NEG SCORE: %f\n", y.neg_score);
    fprintf(fp, "DOCUMENT POLARITY SCORE: %f\n", y.doc_score);
    if(y.nx > 0)
        fprintf(fp, "SENTENCE: [polar] [subj] [joint pos] [joint neg]\n");
    for(i=0; i<y.nx; i++)
    {
        s = y.sentence_scores_polar[i];
        s2 = y.sentence_scores_subj[i];
        fprintf(fp, "%d: %f %f %f %f\n", i, s, s2, s2+s, s2-s);
    }
    fprintf(fp,"\n");
}

void        write_label(FILE *fp, LABEL y)
{
    int i;
    fprintf(fp, "%d ", y.sentiment);
    for(i=0; i<y.n; i++)
        fprintf(fp, "%d ", y.sentence_index[i]);
    fprintf(fp,"\n");
} 

void        free_pattern(PATTERN x) {
    int i;
    for(i=0; i<x.n; i++)
    {
        free_svector(x.sentence_features[i]);
        free_svector(x.sentence_features2[i]); 
    }
    free_svector(x.document_features);
    free(x.sentence_features);
    free(x.sentence_features2);
    x.n = 0;
}

void        free_label(LABEL y) {
    if(y.n > 0)
        free(y.sentence_index);
    if(y.nx > 0)
    {
        free(y.sentence_scores_subj);
        free(y.sentence_scores_polar);
    }
    y.n = 0;
    y.nx = 0;
    y.sentence_scores_subj = NULL;
    y.sentence_scores_polar = NULL;
}

void        free_struct_model(STRUCTMODEL sm) 
{
    /* Frees the memory of model. */
    //if(sm.w) free(sm.w);  /* this is free'd in free_model */
    if(sm.svm_model) free_model(sm.svm_model,1);
    /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
    /* Frees the memory of sample s. */
    int i;
    for(i=0;i<s.n;i++) { 
        free_pattern(s.examples[i].x);
        free_label(s.examples[i].y);
    }
    free(s.examples);
    s.n = 0;
}

void        print_struct_help()
{
    /* Prints a help text that is appended to the common help text of
       svm_struct_learn. */
    /*
    printf("         --* string  -> custom parameters that can be adapted for struct\n");
    printf("                        learning. The * can be replaced by any character\n");
    printf("                        and there can be multiple options starting with --.\n");
    */
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
    /* Parses the command line parameters that start with -- */
    int i;

    sparm->model_loaded = 0;

    for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
        switch ((sparm->custom_argv[i])[2]) 
        { 
            case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
            case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
            case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
            default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
                     exit(0);
        }
    }
}

void        print_struct_help_classify()
{
    /* Prints a help text that is appended to the common help text of
       svm_struct_classify. */
    /*
    printf("         --* string -> custom parameters that can be adapted for struct\n");
    printf("                       learning. The * can be replaced by any character\n");
    printf("                       and there can be multiple options starting with --.\n");
    */
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
    /* Parses the command line parameters that start with -- for the
       classification module */
    int i;

    sparm->model_loaded = 1;

    for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
        switch ((sparm->custom_argv[i])[2]) 
        { 
            /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
            default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
                     exit(0);
        }
    }
}

