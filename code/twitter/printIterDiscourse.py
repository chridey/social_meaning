import os

#markers = {i.replace('_0','').replace('_1','') for i in os.listdir() if i.startswith('tokens_') and '_all_' not in i}
markers = {i.replace('_0','').replace('_1','') for i in os.listdir() if i.startswith('new_tokens_')}

for marker in sorted(markers):
    try:
        with open(marker + '_0') as f:
            wc1 = sum(1 for i in f)
    except IOError:
        continue
    try:
        with open(marker + '_1') as f:
            wc2 = sum(1 for i in f)
    except IOError:
        continue
    #print(marker, wc1, wc2)
    if (0):
        #print('/home/chidey/PDTB/scikit/bin/python3.2 ./trainDiscourseSentiment.py {0} {1} --numTest 5000 --loadModel all_model'.format(marker, min(wc1, wc2)))
        print('/home/chidey/PDTB/scikit/bin/python3.2 ./trainDiscourseSentiment.py {0} {1} --numTest 5000 --tokenizer pair --saveModel model_{0}'.format(marker, min(wc1, wc2)))
    elif (0):
        print('/home/chidey/PDTB/scikit/bin/python3.2 ./trainDiscourseSentiment.py {0} {1} --numTest 5000 --corenlp'.format(marker, min(wc1, wc2)))
    elif (0):
        print('java -cp "lib/*" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -file ../social_meaning/corenlp_test_{0} > ../social_meaning/corenlp_test_{0}_prediction'.format(marker))
    else:
        if wc1 and wc2:
            trainCount = min(wc1, wc2)
            testCount = min(2500, int(trainCount/2))
            print('head -n {} {}_0 >>new_train_0'.format(trainCount-testCount, marker))
            print('head -n {} {}_1 >>new_train_1'.format(trainCount-testCount, marker))
            print('head -n {} {}_0 | tail -n {} >>new_test_0'.format(trainCount, marker, testCount))
            print('head -n {} {}_1 | tail -n {} >>new_test_1'.format(trainCount, marker, testCount))
