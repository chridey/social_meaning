session=$1

cp ../baseline_spam/iterSVM.sh .
ln -s hidden_vars_$session\_0_trainfile hidden_vars_opinion_finder                              
ln -s validatefile_$session\_0 validatefile                                                     
ln -s testfile_$session\_0 testfile                                                             
ln -s trainfile_$session\_0 trainfile                                                           
bash iterSVM.sh 1000 70 3 2