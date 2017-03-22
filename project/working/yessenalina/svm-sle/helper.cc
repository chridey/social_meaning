#include <algorithm>
#include <vector>
#include <map>
#include <deque>
#include <set>
#include <bitset>
#include <iostream>
#include "helper.h"

extern "C" {
#include "./svm_light/svm_common.h"
}

using namespace std;

//
// DECLARE INTERNAL HELPER FUNCTIONS
//


//compares the feature IDs of a WORD object
//useful for sorting 
bool compare_WORD(WORD first, WORD second);

//sentence_scores is the output
//this function isn't really used anymore
void compute_sentence_scores(PATTERN x, STRUCTMODEL *sm, vector< pair<double,int> > & sentence_scores);

//sentence_scores_pos and sentence_scores_neg are the output
//computes the per-sentence joint positive and negative scores
//joint positive is subjectivity + polarity
//joint negative is subjectivity - polarity 
//each entry is represented as a <score,index> paire where score is the score of the sentence at that index 
//scores are then sorted in ascending order
void compute_sentence_scores2(PATTERN x, STRUCTMODEL *sm, vector< pair<double,int> > & sentence_scores_pos, vector< pair<double,int> > & sentence_scores_neg);

//sentence_scores_polar and sentence_scores_subj are the output
//both output objects MUST BE PRE-ALLOCATED
//computes the per-sentence polarity and subjectivity scores
void compute_sentence_scores3(PATTERN x, STRUCTMODEL *sm, double * sentence_scores_polar, double * sentence_scores_subj);

//pos_start, pos_end, neg_start, neg_end, pos_score, neg_score are the output
//this is used only in contiguous window mode 
void find_best_pos_neg(int n, int max_n, double *polar, double *subj, int *pos_start, int *pos_end, int *neg_start, int *neg_end, double *pos_score, double *neg_score);

//Returns a SVECTOR* object that sums up a vector of SVECTOR* objects
//input -- the array SVECTOR* objects to sum up
//n -- the number of SVECTOR* objects
//factor -- the multiplicative factor to multiply each component by
//normalize -- set to 1 if want to normalize output to unit 2-norm.  (default is 1, but this is often turned off, i.e., set to 0)
//offset -- offset the feature ids of the output
//index -- an array of indexes of which SVECTORS of input to sum up.  Can leave as NULL if summing over all of input
//ind_n -- number of indexes in index.  Can leave as 0 if summing over all of input
SVECTOR *sum_svectors(SVECTOR **input, int n, double factor, int normalize=1, long offset=0, int *index=NULL, int ind_n = 0);

//Returns a SVECTOR* object that adds first and second
//second_factor -- multiples second by second_factor prior to adding
//second_offset -- offsets the feature ids of second by second_offset prior to adding
SVECTOR *add_svectors(SVECTOR *first, SVECTOR* second, double second_factor=1, long second_offset=0);

//Returns a SVECTOR* object that is the maximum of the inputted SVECTOR* objects
//input -- the array SVECTOR* object inputs
//n -- the number of inputs
//normalize -- set to 1 if want to normalize output to unit 2-norm.  (default is 1, but this is often turned off, i.e., set to 0)
//offset -- offset the feature ids of the output
//index -- an array of indexes of which SVECTORS of input to consider.  Can leave as NULL if considering all of input
//ind_n -- number of indexes in index.  Can leave as 0 if considering all of input
SVECTOR *max_svectors(SVECTOR **input, int n, int normalize=1, long offset=0, int *index=NULL, int ind_n = 0);

//Returns an empty SVECTOR* object
SVECTOR *create_empty_svector();

//concatenates two feature vectors into one feature vector
//assumes all the features of second comes after first (i.e., no feature overlap)
SVECTOR *concat_svectors(SVECTOR *first, SVECTOR *second, double second_factor=1);

//creates an SVECTOR* object representing the feature vector in vector<WORD> form
//offset -- offset the feature ids of the output
SVECTOR *create_svector_cc(vector<WORD> &v, long offset=0);

//
// END DECLARE INTERNAL HELPER FUNCTIONS
//

// duplicates the features of each sentence
// the second set of features is offsetted
void process_sentence_features(SAMPLE *sample, long offset)
{
    for(int i=0; i<sample->n; i++)
    {
        for(int j=0; j<sample->examples[i].x.n; j++)
        {
            vector<WORD> all_words;
            int ind = 0;
            SVECTOR *s = sample->examples[i].x.sentence_features[j];
            while(s->words[ind].wnum)
            {
                WORD w = s->words[ind];
                all_words.push_back(w);
                w.wnum += offset;
                all_words.push_back(w);
                ind++;
            }
            sort(all_words.begin(),all_words.end(), compare_WORD);
            free_svector(sample->examples[i].x.sentence_features[j]);
            sample->examples[i].x.sentence_features[j] = create_svector_cc(all_words);
        }
    }
}


// adds offset to the feature IDs of the sentence-level subjectivity features
void process_sentence_features2(SAMPLE *sample, long offset)
{
    for(int i=0; i<sample->n; i++)
    {
        for(int j=0; j<sample->examples[i].x.n; j++)
        {
            vector<WORD> all_words;
            int ind = 0;
            SVECTOR *s = sample->examples[i].x.sentence_features2[j];
            while(s->words[ind].wnum)
            {
                s->words[ind].wnum += offset;
                ind++;
            }
        }
    }
}

// adds offset to the feature IDs of the document-level features
void process_document_features(SAMPLE *sample, long offset)
{
    for(int i=0; i<sample->n; i++)
    {
       int ind = 0;
       SVECTOR *s = sample->examples[i].x.document_features;
       while(s->words[ind].wnum)
       {
           s->words[ind].wnum += offset;
           ind++;
       }
    }
}


// reads in inputs from a data file
SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {

    FILE *fp = fopen(filename,"r");
    if (fp==NULL) {
        printf("Cannot open input file %s!\n", filename); 
        exit(1);
    }

    if(sparm->model_loaded == 0)
    {
        sparm->max_feature_key = 0;
        sparm->max_sentence_feature_key = 0;
        sparm->max_doc_feature_key = 0;
    }

    vector< vector< vector<WORD> > > sentence_features;
    vector< vector< vector<WORD> > > sentence_features2;
    vector< vector<WORD> >  doc_features;
    vector<int> labels;

    int num_sentences, label;
    // reading the label of doc (+1 or -1)
    fscanf(fp, "%d", &label);

    while(true)
    {
        FNUM key;
        FVAL value;
        vector< vector<WORD> > sentences;
        vector< vector<WORD> > sentences2;

        // reading the number of sentences in doc
        if(fscanf(fp, "%d", &num_sentences) != 1) 
            break;

        int dummy, read;
        fscanf(fp, "%d", &dummy);


        // one line of features for each sentence
        for(int i=0; i<num_sentences; i++)
        {
            vector<WORD> sentence;
            vector<WORD> sentence2;

            // reading valid key:value pair of polarity features
            while((read=fscanf(fp, "%d:%f", &key, &value))==2)
            {
                if(sparm->model_loaded == 1 && key > sparm->max_feature_key)
                    continue;

                WORD w;
                w.wnum = key;
                w.weight = value;
                sentence.push_back(w);
                if(key > sparm->max_feature_key) 
                    sparm->max_feature_key = key;
            }

            sentences.push_back(sentence);

            // reading key:value pair of subjectivity features
            if(read == 0)
            {
                while(fscanf(fp, " S%d:%f", &key, &value)==2)
                {
                    if(sparm->model_loaded == 1 && key > sparm->max_sentence_feature_key)
                        continue;
                    
                    WORD w;
                    w.wnum = key;
                    w.weight = value;
                    sentence2.push_back(w);
                    if(key > sparm->max_sentence_feature_key) 
                        sparm->max_sentence_feature_key = key;
                }

                fscanf(fp, "%d", &dummy);
            }

            sentences2.push_back(sentence2);
        }

        sentence_features.push_back(sentences);
        sentence_features2.push_back(sentences2);
        labels.push_back(label);

        //assert(key == num_sentences);
        vector<WORD> doc_feat;

        // reading valid key:value pair of document-level features
        while((read=fscanf(fp, "%d:%f", &key, &value))==2)
        {
            if(sparm->model_loaded == 1 && key > sparm->max_doc_feature_key)
                continue;

            WORD w;
            w.wnum = key;
            w.weight = value;
            doc_feat.push_back(w);
            if(key > sparm->max_doc_feature_key) 
                sparm->max_doc_feature_key = key;
        }
        doc_features.push_back(doc_feat);

        label = key;
    }

    fclose(fp);

    long n = labels.size();

    SAMPLE sample;
    sample.n = n;
    sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*sample.n);

    for(int i=0; i<sample.n; i++)
    {
        sample.examples[i].y.n=0;
        sample.examples[i].y.sentence_scores_polar = NULL;
        sample.examples[i].y.sentence_scores_subj = NULL;
        sample.examples[i].y.sentence_index = NULL;

        sample.examples[i].y.sentiment = labels[i];
        int num_sent = sentence_features[i].size();
        sample.examples[i].x.n = num_sent;
        sample.examples[i].x.sentence_features = (SVECTOR**)malloc(sizeof(SVECTOR*)*num_sent);
        sample.examples[i].x.sentence_features2 = (SVECTOR**)malloc(sizeof(SVECTOR*)*num_sent);

        for(int k=0; k<num_sent; k++)
        {
            sample.examples[i].x.sentence_features[k] = create_svector_cc(sentence_features[i][k]);
            sample.examples[i].x.sentence_features2[k] = create_svector_cc(sentence_features2[i][k]);
        }

        sample.examples[i].x.document_features = create_svector_cc(doc_features[i]);
    }

    return sample;
}


// computes the Psi joint feature vector
SVECTOR *psi_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    double r_factor = extract_norm(y.n, sm->norm_mode);
    double factor = r_factor * (double)y.sentiment;
    double factor2 = r_factor;

    SVECTOR *ans0, *ans1, *ans2, *ans;

    if(sm->feature_mode == MODE_ORIG)
    {
        ans0 = sum_svectors(x.sentence_features, x.n, factor, 0, 0, y.sentence_index, y.n);
        ans1 = sum_svectors(x.sentence_features2, x.n, factor2, 0, 0, y.sentence_index, y.n);
        ans = concat_svectors(ans0,ans1);
        free_svector(ans0);
        free_svector(ans1);

    }
    else if(sm->feature_mode == MODE_FLAT)
    {
        ans0 = create_empty_svector();
        ans = concat_svectors(ans0, x.document_features, 0.5*(double)y.sentiment);
        free_svector(ans0);

        //Note that we multiply the document-level scores by 0.5, this is so that using 
        //only document-level scores will reduce properly to a standard SVM
    }
    else //if(sm->feature_mode == MODE_ORIG_AND_FLAT || sm->feature_mode == MODE_FEAT_SMOOTH)
    {
        ans0 = sum_svectors(x.sentence_features, x.n, factor, 0, 0, y.sentence_index, y.n);
        ans1 = sum_svectors(x.sentence_features2, x.n, factor2, 0, 0, y.sentence_index, y.n);
        ans2 = add_svectors(ans0,ans1);

        ans = add_svectors(ans2, x.document_features, 0.5*(double)y.sentiment);
        //Note that we multiply the document-level scores by 0.5, this is so that using 
        //only document-level scores will reduce properly to a standard SVM

        free_svector(ans0);
        free_svector(ans1);
        free_svector(ans2);
    }

    return ans;
}

//initializes various parameters
CONSTSET init_struct_constraints_helper(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    CONSTSET c;

    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;

    char dummy[1000];

    // this logic runs if a prior model was specified
    if(sparm->model_prior_file != NULL)
    {
        FILE *modelfl;
        int fnum;
        double fweight;
        vector<WORD> lhs;

        printf("Loading model prior...");

        modelfl = fopen(sparm->model_prior_file,"r");
        if (modelfl==NULL) {
            printf("Cannot open model prior file %s for input!", sparm->model_prior_file);
            return c;
        }

        while (!feof(modelfl)) 
        {
            if (fscanf(modelfl, "#%s\n", dummy)> 0)
                continue;

            if(fscanf(modelfl, "%d:%lf", &fnum, &fweight) == 2)
            {
                if(fweight != 0)// && fnum <= sm->sizePsi)
                {
                    if(fnum <= sm->sizePsi)
                    {
                        WORD w;
                        w.wnum = fnum;
                        w.weight = fweight/sparm->C;  // this is not a training example, so we divide by C
                        lhs.push_back(w);
                    }
                    else
                    {
                        printf("WARNING: model prior has feature id (%d) greater than\n max feature id in data (%ld) -- feature %d is ignored\n", fnum, sm->sizePsi, fnum);
                    }
                }
            }
        }

        fclose(modelfl); 

        sort(lhs.begin(),lhs.end(), compare_WORD);
        SVECTOR *s = create_svector_cc(lhs);
        c.m = 1;
        c.lhs = (DOC **)malloc(sizeof(DOC *));
        c.lhs[0] = create_example(0,0,sample.n+1,1,s);
        c.rhs = (double *)malloc(sizeof(double));
        c.rhs[0] = 99999999;

        //the regularizing relative to a prior is equivalent to minimizing
        // |w|^2/2 - <w,w0> + (C/N)sum_{i=1}^{N} Xi_i
        //where w0 is the prior model, and Xi are the slack variables
        //this can be implemented by introducing a new slack constraint:
        // Xi_0 >= T - <w,w0>
        //where T is some really large value (in our case T=99999999)

        printf("done\n");
    }

    return(c);
}

// primary logic for implementing Line 7 of Algorithm 2 in paper
LABEL infer_hidden_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    LABEL h;
    h.nx = 0;
    h.sentiment = y.sentiment;
    
    if(sm->feature_mode == MODE_FLAT)
    {
        h.n = 0;
        if(sparm->classify_verbose == 1)
        {
            h.pos_score = 0;
            h.neg_score = 0;
            h.doc_score = 0;
        }
        return h;
    }

    //extraction max size
    int max_n = max_extract_size(x.n, sm->latent_size_mode);    

    h.n = max_n;
    h.sentence_index = (int *)malloc(sizeof(int)*max_n);

    if(sm->window_mode == 1)
    {
        int pos_start, pos_end, neg_start, neg_end;
        double pos_score, neg_score;
        h.nx = x.n;
        h.sentence_scores_polar = (double *)malloc(sizeof(double)*x.n);
        h.sentence_scores_subj = (double *)malloc(sizeof(double)*x.n);
        compute_sentence_scores3(x,sm,h.sentence_scores_polar,h.sentence_scores_subj);

        find_best_pos_neg(h.nx, max_n, h.sentence_scores_polar, h.sentence_scores_subj, &pos_start, &pos_end, &neg_start, &neg_end, &pos_score, &neg_score);
        
        if(y.sentiment == 1)
        {
            for(int i = 0; i < pos_end-pos_start; i++)
                h.sentence_index[i] = pos_start + i;
            for(int i = pos_end-pos_start; i < max_n; i++)
                h.sentence_index[i] = -1;
        }
        else
        {
            for(int i = 0; i < neg_end-neg_start; i++)
                h.sentence_index[i] = neg_start + i;
            for(int i = neg_end-neg_start; i < max_n; i++)
                h.sentence_index[i] = -1;
        }
    }
    else
    {
        vector< pair<double,int> > sentence_scores_pos;
        vector< pair<double,int> > sentence_scores_neg;
        compute_sentence_scores2(x,sm,sentence_scores_pos,sentence_scores_neg);

        //computing positive score
        int pos_size = 1;
        while(pos_size <= max_n && sentence_scores_pos[x.n-pos_size].first > 0)
            pos_size++;
        pos_size--;

        //computing negative score
        int neg_size = 1;
        while(neg_size <= max_n && sentence_scores_neg[x.n-neg_size].first > 0)
            neg_size++;
        neg_size--;

        if(y.sentiment == 1)
        {
            for(int j =0; j<pos_size; j++)
                h.sentence_index[j] = sentence_scores_pos[x.n-j-1].second;
            for(int j=pos_size; j<max_n; j++)
                h.sentence_index[j] = -1;
        }
        else
        {
            for(int j =0; j<neg_size; j++)
                h.sentence_index[j] = sentence_scores_neg[x.n-j-1].second;
            for(int j=neg_size; j<max_n; j++)
                h.sentence_index[j] = -1;
        }
    }

    return h;
}

// classifies the document-level sent and also extracts the best supporting sentences
LABEL classify_struct_example_helper(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    LABEL y;
    y.nx = 0;

    //will be 0 if ignoring document features (i.e., MODE_FLAT)
    double doc_score = sprod_ns(sm->w, x.document_features)*0.5;
    //Note that we multiply the document-level scores by 0.5, this is so that using 
    //only document-level scores will reduce properly to a standard SVM

    //doc only features
    if(sm->feature_mode == MODE_FLAT)
    {
        y.n = 0;
        y.sentiment = -1;
        if(doc_score > 0)
            y.sentiment = 1;
        if(sparm->classify_verbose == 1)
        {
            y.pos_score = 0;
            y.neg_score = 0;
            y.doc_score = doc_score;
        }
        return y;
    }

    //extraction max size
    int max_n = max_extract_size(x.n, sm->latent_size_mode);    

    //normalization factor
    double factor = extract_norm(max_n, sm->norm_mode);

    y.n = max_n;
    y.sentence_index = (int *)malloc(sizeof(int)*max_n);

    if(sm->window_mode == 1)
    {
        int pos_start, pos_end, neg_start, neg_end;
        double pos_score, neg_score;
        y.nx = x.n;
        y.sentence_scores_polar = (double *)malloc(sizeof(double)*x.n);
        y.sentence_scores_subj = (double *)malloc(sizeof(double)*x.n);
        compute_sentence_scores3(x,sm,y.sentence_scores_polar,y.sentence_scores_subj);

        //extraction max size
        int max_n = max_extract_size(x.n, sm->latent_size_mode);    

        //finds the best positive and negative windows
        find_best_pos_neg(y.nx, max_n, y.sentence_scores_polar, y.sentence_scores_subj, &pos_start, &pos_end, &neg_start, &neg_end, &pos_score, &neg_score);

        y.pos_score = pos_score*factor;
        y.neg_score = neg_score*factor;
        y.doc_score = doc_score;

        //predict 1 if positive score greater than negative score
        if(y.pos_score + y.doc_score > y.neg_score - y.doc_score)
        {
            y.sentiment = 1;
            for(int i = 0; i < pos_end-pos_start; i++)
                y.sentence_index[i] = pos_start + i;
            for(int i = pos_end-pos_start; i < max_n; i++)
                y.sentence_index[i] = -1;
        }
        else
        {
            y.sentiment = -1;
            for(int i = 0; i < neg_end-neg_start; i++)
                y.sentence_index[i] = neg_start + i;
            for(int i = neg_end-neg_start; i < max_n; i++)
                y.sentence_index[i] = -1;
        }
    }
    else
    {
        double pos_score = doc_score;
        double neg_score = -doc_score;

        vector< pair<double,int> > sentence_scores_pos;
        vector< pair<double,int> > sentence_scores_neg;
        compute_sentence_scores2(x,sm,sentence_scores_pos,sentence_scores_neg);

        //computing positive score
        int pos_size = 1;
        while(pos_size <= max_n && sentence_scores_pos[x.n-pos_size].first > 0)
        {
            pos_score += sentence_scores_pos[x.n-pos_size].first * factor;
            pos_size++;
        }
        pos_size--;

        //computing negative score
        int neg_size = 1;
        while(neg_size <= max_n && sentence_scores_neg[x.n-neg_size].first > 0)
        {
            neg_score += sentence_scores_neg[x.n-neg_size].first * factor;
            neg_size++;
        }
        neg_size--;

        if(sparm->classify_verbose == 1)
        {
            y.pos_score = pos_score - doc_score;
            y.neg_score = neg_score + doc_score;
            y.doc_score = doc_score;


            if(y.nx == 0)
            {
                y.sentence_scores_polar = (double *)malloc(sizeof(double)*x.n);
                y.sentence_scores_subj = (double *)malloc(sizeof(double)*x.n);
            }

            y.nx = x.n;

            compute_sentence_scores3(x,sm,y.sentence_scores_polar,y.sentence_scores_subj);
        }

        //predicting label based on which score is higher
        if(pos_score > neg_score)
        {
            y.sentiment = 1;
            for(int j =0; j<pos_size; j++)
                y.sentence_index[j] = sentence_scores_pos[x.n-j-1].second;
            for(int j=pos_size; j<max_n; j++)
                y.sentence_index[j] = -1;
        }
        else
        {
            y.sentiment = -1;
            for(int j =0; j<neg_size; j++)
                y.sentence_index[j] = sentence_scores_neg[x.n-j-1].second;
            for(int j=neg_size; j<max_n; j++)
                y.sentence_index[j] = -1;
        }
    }

    return y;
}

// finds the most violated constraint in standard cutting plane training of Structural SVMs
LABEL find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, 
        STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    /*
       Finds the most violated constraint (loss-augmented inference), i.e.,
       computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
       The output (ybar,hbar) are stored at location pointed by 
       pointers *ybar and *hbar. 
       */
    LABEL ybar;

    ybar.nx = 0;

    //will be 0 if ignoring document features
    double doc_score = sprod_ns(sm->w, x.document_features)*0.5;
    //Note that we multiply the document-level scores by 0.5, this is so that using 
    //only document-level scores will reduce properly to a standard SVM

    //doc only features
    if(sm->feature_mode == MODE_FLAT)
    {
        ybar.n = 0;
        ybar.sentiment = -1;
        if(y.sentiment == 1 && 2.0*doc_score - 1 > 0)
            ybar.sentiment = 1;
        if(y.sentiment == -1 && 2.0*doc_score + 1 > 0)
            ybar.sentiment = 1;
        if(sparm->classify_verbose == 1)
        {
            ybar.pos_score = 0;
            ybar.neg_score = 0;
            ybar.doc_score = doc_score;
        }
        return ybar;
    }

    //extract max size
    int max_n = max_extract_size(x.n, sm->latent_size_mode);    

    //normalization factor
    double factor = extract_norm(max_n, sm->norm_mode);

    ybar.n = max_n;
    ybar.sentence_index = (int*)malloc(sizeof(int)*max_n);

    if(sm->window_mode == 1)
    { 
        int pos_start, pos_end, neg_start, neg_end;
        double pos_score, neg_score;
        ybar.nx = x.n;
        ybar.sentence_scores_polar = (double *)malloc(sizeof(double)*x.n);
        ybar.sentence_scores_subj = (double *)malloc(sizeof(double)*x.n);
        compute_sentence_scores3(x,sm,ybar.sentence_scores_polar,ybar.sentence_scores_subj);

        //extraction max size
        int max_n = max_extract_size(x.n, sm->latent_size_mode);    

        //finds the best positive and negative windows
        find_best_pos_neg(ybar.nx, max_n, ybar.sentence_scores_polar, ybar.sentence_scores_subj, &pos_start, &pos_end, &neg_start, &neg_end, &pos_score, &neg_score);

        ybar.pos_score = pos_score*factor;
        ybar.neg_score = neg_score*factor;
        ybar.doc_score = doc_score;
        
        //augmenting by loss
        if(y.sentiment == 1)
            ybar.neg_score += 1;
        else
            ybar.pos_score += 1;
        
        if(ybar.pos_score + ybar.doc_score > ybar.neg_score - ybar.doc_score
                || (ybar.pos_score + ybar.doc_score == ybar.neg_score  -ybar.doc_score && y.sentiment == -1))
        {
            ybar.sentiment = 1;
            for(int i = 0; i < pos_end-pos_start; i++)
                ybar.sentence_index[i] = pos_start + i;
            for(int i = pos_end-pos_start; i < max_n; i++)
                ybar.sentence_index[i] = -1;
        }
        else
        {
            ybar.sentiment = -1;
            for(int i = 0; i < neg_end-neg_start; i++)
                ybar.sentence_index[i] = neg_start + i;
            for(int i = neg_end-neg_start; i < max_n; i++)
                ybar.sentence_index[i] = -1;
        }
    }
    else
    {
        double pos_score = doc_score;
        double neg_score = -doc_score;

        //augmenting by loss
        if(y.sentiment == 1)
            neg_score += 1;
        else
            pos_score += 1;

        vector< pair<double,int> > sentence_scores_pos;
        vector< pair<double,int> > sentence_scores_neg;
        compute_sentence_scores2(x,sm,sentence_scores_pos,sentence_scores_neg);

        //computing positive score
        int pos_size = 1;
        while(pos_size <= max_n && sentence_scores_pos[x.n-pos_size].first > 0)
        {
            pos_score += sentence_scores_pos[x.n-pos_size].first * factor;
            pos_size++;
        }
        pos_size--;

        //computing negative score
        int neg_size = 1;
        while(neg_size <= max_n && sentence_scores_neg[x.n-neg_size].first > 0)
        {
            neg_score += sentence_scores_neg[x.n-neg_size].first * factor;
            neg_size++;
        }
        neg_size--;

        if(sparm->classify_verbose == 1)
        {
            ybar.pos_score = pos_score - doc_score;
            ybar.neg_score = neg_score + doc_score;
            ybar.doc_score = doc_score;


            if(y.nx == 0)
            {
                ybar.sentence_scores_polar = (double *)malloc(sizeof(double)*x.n);
                ybar.sentence_scores_subj = (double *)malloc(sizeof(double)*x.n);
            }

            ybar.nx = x.n;

            compute_sentence_scores3(x,sm,ybar.sentence_scores_polar,ybar.sentence_scores_subj);
        }

        ybar.n = max_n;
        ybar.sentence_index = (int*)malloc(sizeof(int)*max_n);
        if(pos_score > neg_score || (pos_score == neg_score && y.sentiment == -1))
        {
            ybar.sentiment = 1;
            for(int j =0; j<pos_size; j++)
                ybar.sentence_index[j] = sentence_scores_pos[x.n-j-1].second;
            for(int j=pos_size; j<max_n; j++)
                ybar.sentence_index[j] = -1;
        }
        else
        {
            ybar.sentiment = -1;
            for(int j =0; j<neg_size; j++)
                ybar.sentence_index[j] = sentence_scores_neg[x.n-j-1].second;
            for(int j=neg_size; j<max_n; j++)
                ybar.sentence_index[j] = -1;
        }
    }

    return ybar;
}

//initializes the latent variables
//requires reading from the latent_file inputted during learning
void init_latent_helper(SAMPLE *sample, STRUCTMODEL *sm, char *file)
{
    for(int i=0; i<sample->n; i++)
    {
        sample->examples[i].y.n=0;
        sample->examples[i].y.sentence_scores_polar = NULL;
        sample->examples[i].y.sentence_scores_subj = NULL;
    }

    if(sm->feature_mode == MODE_FLAT)
        return;

    FILE *fp = fopen(file,"r");
    if (fp==NULL) {
        printf("Cannot open latent variable initialization file %s!\n", file); 
        exit(1);
    }

    int num_extract, ind;
    for(int i=0; i<sample->n; i++)
    {
        fscanf(fp, "%d", &num_extract);

        sample->examples[i].y.n=num_extract;
        sample->examples[i].y.sentence_index = (int *)malloc(sizeof(int)*num_extract);

        for(int j=0; j<num_extract; j++)
        {
            fscanf(fp, "%d", &ind);
            sample->examples[i].y.sentence_index[j] = ind;
        }
    }
}


//
// DEFINE INTERNAL HELPER FUNCTIONS
//

bool compare_WORD(WORD first, WORD second)
{
    if(first.wnum < second.wnum)
        return true;
    return false;
}

//sentence_scores is the output
void compute_sentence_scores(PATTERN x, STRUCTMODEL *sm, vector< pair<double,int> > & sentence_scores)
{
    //scoring individual sentences
    for(int i=0; i<x.n; i++)
    {
        pair<double,int> score;
        score.first = sprod_ns(sm->w,x.sentence_features[i]);
        score.second = i;
        sentence_scores.push_back(score);
    }
    sort(sentence_scores.begin(),sentence_scores.end());
}

//sentence_scores_pos and sentence_scores_neg are the output
void compute_sentence_scores2(PATTERN x, STRUCTMODEL *sm, vector< pair<double,int> > & sentence_scores_pos,vector< pair<double,int> > & sentence_scores_neg)
{
    //scoring individual sentences
    for(int i=0; i<x.n; i++)
    {
        pair<double,int> score_pos;
        pair<double,int> score_neg;
        double s = sprod_ns(sm->w,x.sentence_features[i]);
        double s2 = sprod_ns(sm->w,x.sentence_features2[i]);
        score_pos.first = s2 + s;
        score_neg.first = s2 - s;
        score_pos.second = i;
        score_neg.second = i;
        sentence_scores_pos.push_back(score_pos);
        sentence_scores_neg.push_back(score_neg);
    }
    sort(sentence_scores_pos.begin(),sentence_scores_pos.end());
    sort(sentence_scores_neg.begin(),sentence_scores_neg.end());
}

//sentence_scores_polar and sentence_scores_subj are the output
void compute_sentence_scores3(PATTERN x, STRUCTMODEL *sm, double * sentence_scores_polar, double * sentence_scores_subj)
{
    //scoring individual sentences
    for(int i=0; i<x.n; i++)
    {
        pair<double,int> score_pos;
        pair<double,int> score_neg;
        double s = sprod_ns(sm->w,x.sentence_features[i]);
        double s2 = sprod_ns(sm->w,x.sentence_features2[i]);
        sentence_scores_polar[i] = s;
        sentence_scores_subj[i] = s2;
    }
}

// pos_start, pos_end, neg_start, neg_end, pos_score, neg_score are the output
void find_best_pos_neg(int n, int max_n, double *polar, double *subj, int *pos_start, int *pos_end, int *neg_start, int *neg_end, double *pos_score, double *neg_score)
{
    double curr_neg_score = 0;
    double curr_pos_score = 0;

    *pos_start = -1;
    *neg_start = -1;
    *pos_end = -1;
    *neg_end = -1;
    *pos_score = 0;
    *neg_score = 0;


    for(int i = 0; i< n; i++)
    {
        curr_neg_score = 0;
        curr_pos_score = 0;

        for(int k = i; k < i + max_n && k < n; k++)
        {
            curr_pos_score += subj[k] + polar[k];
            curr_neg_score += subj[k] - polar[k];

            if(curr_pos_score > *pos_score)
            {
                *pos_start = i;
                *pos_end = k+1;
                *pos_score = curr_pos_score;
            }

            if(curr_neg_score > *neg_score)
            {
                *neg_start = i;
                *neg_end = k+1;
                *neg_score = curr_neg_score;
            }
        }
    }
}


SVECTOR *sum_svectors(SVECTOR **input, int n, double factor, int normalize, long offset, int *index, int ind_n)
{
    map<FNUM,FVAL> all_words;
    double count = 0;
    if(ind_n > 0)
    {
        //printf("summing using index\n");
        //printf("num: %d\n", ind_n);
        for(int i=0; i<ind_n; i++)
        {
            int h_ind = index[i];
            //printf("%d %d\n",i,h_ind);fflush(stdout);
            if(h_ind >= 0)
            {
                SVECTOR *s = input[h_ind];
                int ind = 0;
                while(s->words[ind].wnum)
                {
                    all_words[s->words[ind].wnum + offset] += s->words[ind].weight;
                    count += s->words[ind].weight;
                    ind++;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<n; i++)
        {
            SVECTOR *s = input[i];
            int ind = 0;
            while(s->words[ind].wnum)
            {
                all_words[s->words[ind].wnum + offset] += s->words[ind].weight;
                count += s->words[ind].weight;
                ind++;
            }
        }
    }

    vector<WORD> all_words2;

    double factor2 = factor;
    if(normalize == 1)
        factor2 /= sqrt(count);

    for(map<FNUM,FVAL>::iterator it = all_words.begin(); it != all_words.end(); it++)
    {
        WORD w;
        w.wnum = (*it).first;
        w.weight = (*it).second * factor2;
        all_words2.push_back(w);
    }

    sort(all_words2.begin(),all_words2.end(), compare_WORD);

    return create_svector_cc(all_words2);
} 

SVECTOR *max_svectors(SVECTOR **input, int n, int normalize, long offset, int *index, int ind_n)
{
    map<FNUM,FVAL> all_words;
    double count = 0;
    if(ind_n > 0)
    {
        for(int i=0; i<ind_n; i++)
        {
            int h_ind = index[i];
            if(h_ind >= 0)
            {
                SVECTOR *s = input[h_ind];
                int ind = 0;
                while(s->words[ind].wnum)
                {
                    int off_ind = s->words[ind].wnum + offset;
                    if(s->words[ind].weight > all_words[off_ind])
                    {
                        count += s->words[ind].weight - all_words[off_ind];
                        all_words[off_ind] = s->words[ind].weight;
                    }
                    ind++;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<n; i++)
        {
            SVECTOR *s = input[i];
            int ind = 0;
            while(s->words[ind].wnum)
            {
                int off_ind = s->words[ind].wnum + offset;
                if(s->words[ind].weight > all_words[off_ind])
                {
                    count += s->words[ind].weight - all_words[off_ind];
                    all_words[s->words[ind].wnum + offset] = s->words[ind].weight;
                }
                ind++;
            }
        }
    }

    vector<WORD> all_words2;

    double factor = 1.0;
    if(normalize == 1)
        factor = 1.0/sqrt(count);
    for(map<FNUM,FVAL>::iterator it = all_words.begin(); it != all_words.end(); it++)
    {
        WORD w;
        w.wnum = (*it).first;
        w.weight = (*it).second * factor;
        all_words2.push_back(w);
    }

    sort(all_words2.begin(),all_words2.end(), compare_WORD);

    return create_svector_cc(all_words2);
} 

SVECTOR *add_svectors(SVECTOR *first, SVECTOR* second, double second_factor, long second_offset)
{
    map<FNUM,FVAL> all_words;
    int ind = 0;
    while(first->words[ind].wnum)
    {
        all_words[first->words[ind].wnum] += first->words[ind].weight;
        ind++;
    }
    ind = 0;
    while(second->words[ind].wnum)
    {
        int ind2 = second->words[ind].wnum + second_offset;
        all_words[ind2] += second->words[ind].weight * second_factor;
        ind++;
    }


    vector<WORD> all_words2;

    for(map<FNUM,FVAL>::iterator it = all_words.begin(); it != all_words.end(); it++)
    {
        WORD w;
        w.wnum = (*it).first;
        w.weight = (*it).second;
        all_words2.push_back(w);
    }

    sort(all_words2.begin(),all_words2.end(), compare_WORD);

    return create_svector_cc(all_words2);
}

SVECTOR *concat_svectors(SVECTOR *first, SVECTOR *second, double second_factor)
{
    vector<WORD> all_words;

    int ind = 0;
    while(first->words[ind].wnum)
    {
        all_words.push_back(first->words[ind]);
        ind++;
    }

    ind = 0;
    while(second->words[ind].wnum)
    {
        WORD w = second->words[ind];
        w.weight *= second_factor;
        all_words.push_back(w);
        ind++;
    }

    return create_svector_cc(all_words);
}

SVECTOR *create_empty_svector()
{
    WORD w;
    w.wnum = 0;
    w.weight = 0;

    SVECTOR *ans = create_svector(&w,"",1);

    return ans;
}


SVECTOR *create_svector_cc(vector<WORD> &v, long offset)
{
    if(v.size() == 0)
        return create_empty_svector();

    WORD *words;
    words = new WORD[v.size()+1];
    copy(v.begin(),v.end(),words);
    for(int i=0; i<(int)v.size(); i++)
        words[i].wnum += offset;
    words[v.size()].wnum = 0;
    words[v.size()].weight = 0;
    SVECTOR *ans = create_svector(words,"",1);
    delete [] words;
    return ans;
}
