c=$1
l=$2

mkdir c$c.l$l

#features=$3
#session=$4

echo 0
bin/svm_sle_learn -y 3 -v 3 -c $c -l $l trainfile hidden_vars_opinion_finder c$c.l$l/full_model_c$c.l$l >> c$c.l$l/svm_training_log
bin/svm_sle_classify -l trainfile c$c.l$l/full_model_c$c.l$l c$c.l$l/hidden_vars_c$c.l$l.1 >> c$c.l$l/svm_training_log
bin/svm_sle_classify validatefile c$c.l$l/full_model_c$c.l$l c$c.l$l/validate_scores_c$c.l$l >> c$c.l$l/svm_classify_log

for i in `seq 5`
do
echo $i
bin/svm_sle_learn -y 3 -v 3 -c $c -l $l trainfile c$c.l$l/hidden_vars_c$c.l$l.$i c$c.l$l/full_model_c$c.l$l.$i >> c$c.l$l/svm_training_log
bin/svm_sle_classify -l trainfile c$c.l$l/full_model_c$c.l$l.$i c$c.l$l/hidden_vars_c$c.l$l.$((i+1)) >>c$c.l$l/svm_training_log
bin/svm_sle_classify validatefile c$c.l$l/full_model_c$c.l$l.$i c$c.l$l/validate_scores_c$c.l$l.$i >> c$c.l$l/svm_classify_log
done