session=$1
p=$2
s=$3

echo $p
echo $s

mkdir $session
cd $session
for i in 1 2 3 4 5;
do
~/PDTB/scikit/bin/python3.2 ../generateSLEFormat.py $session$i 320 --polarityFeatures $p  --subjectivityFeatures $s --iterations 10 -l 70 --dataDir /local/nlp/chidey/social_meaning/op_spam_v1.4/experiments_positive/fold$i > $session$i\_log 
fgrep Zero $session$i\_log >> results

#cd $session$i
#bash ../../runIterSVM.sh $session$i
#fgrep Zero $session*/c1000.l70.m3.n2/* > ../results
done

