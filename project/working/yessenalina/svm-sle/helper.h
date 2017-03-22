#ifdef __cplusplus
extern "C" {
#endif
#include "svm_struct_api_types.h"
#include "svm_struct/svm_struct_common.h"


int max_extract_size(int num_sentences, int mode);
double extract_norm(int extract_size, int mode);

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_helper(PATTERN x, LABEL y,  STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LABEL classify_struct_example_helper(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LABEL infer_hidden_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LABEL find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void init_latent_helper(SAMPLE *sample, STRUCTMODEL *sm, char *file);
void process_document_features(SAMPLE *sample, long offset);
void process_sentence_features(SAMPLE *sample, long offset);
void process_sentence_features2(SAMPLE *sample, long offset);

CONSTSET init_struct_constraints_helper(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

#ifdef __cplusplus
}
#endif
