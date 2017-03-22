
#!/bin/bash

if [[ ! ("$#" == 1) ]]; then
    echo ""
    echo ""
    echo  'Usage: sample_scrpit.sh  C-parameter'
    echo ""
    echo "This script runs SVM^{sle} executables for a 5 iterations"
    echo "on 0-th fold of Movie Reviews data."
    echo ''
    echo ''
    exit 1
fi    
 
c=$1

echo $c
e=0.001

rm -rf  _latent _models _results
mkdir _latent
mkdir _models
mkdir _results

for i in  0 # 1 2 3 4 5 6 7 8
do

  ./bin/svm_sle_learn  -c $c   -e $e  ./movieReviews/train.$i.txt  ./hiddenvars_OpinionFinder/train.$i  ./_models/mOF.c.$c.e.$e.iter.0
  ./bin/svm_sle_classify -l ./movieReviews/train.$i.txt   ./_models/mOF.c.$c.e.$e.iter.0  ./_latent/hvOF.c.$c.e.$e.iter.1
     
  ./bin/svm_sle_classify  ./movieReviews/test.0.txt   ./_models/mOF.c.$c.e.$e.iter.0   > ./_results/rOF.test.c.$c.e.$e.full.txt
  grep "Zero/one" ./_results/rOF.test.c.$c.e.$e.full.txt | cut -d' ' -f5  | tr -d % >> ./_results/rOF.test.c.$c.e.$e.iter.0.txt
  
  ./bin/svm_sle_classify  ./movieReviews/valid.$i.txt  ./_models/mOF.c.$c.e.$e.iter.0   > ./_results/rOF.valid.c.$c.e.$e.full.txt
  grep "Zero/one" ./_results/rOF.valid.c.$c.e.$e.full.txt | cut -d' ' -f5  | tr -d % >> ./_results/rOF.valid.c.$c.e.$e.iter.0.txt            
  
  for iter in 1 2 3 4
       do
    ./bin/svm_sle_learn  -c $c  -e $e  ./movieReviews/train.$i.txt  ./_latent/hvOF.c.$c.e.$e.iter.$iter  ./_models/mOF.c.$c.e.$e.iter.$iter  
    ./bin/svm_sle_classify -l ./movieReviews/train.$i.txt   ./_models/mOF.c.$c.e.$e.iter.$iter  ./_latent/hvOF.c.$c.e.$e.iter.$((iter+1))

    ./bin/svm_sle_classify  ./movieReviews/test.0.txt   ./_models/mOF.c.$c.e.$e.iter.$iter   > ./_results/rOF.test.c.$c.e.$e.full.txt
    grep "Zero/one" ./_results/rOF.test.c.$c.e.$e.full.txt |cut -d' ' -f5| tr -d % >> ./_results/rOF.test.c.$c.e.$e.iter.$iter.txt
                             
    ./bin/svm_sle_classify  ./movieReviews/valid.$i.txt  ./_models/mOF.c.$c.e.$e.iter.$iter  > ./_results/rOF.valid.c.$c.e.$e.full.txt
    grep "Zero/one" ./_results/rOF.valid.c.$c.e.$e.full.txt |cut -d' ' -f5| tr -d % >> ./_results/rOF.valid.c.$c.e.$e.iter.$iter.txt
    
  done  
done
