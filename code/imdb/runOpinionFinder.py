from __future__ import print_function

import sys
import os

for fn in os.listdir():
    if fn.endswith('.sentences'):
        fn = fn.replace(' ', '\ ')
        print("java -classpath ./lib/weka.jar:./lib/stanford-postagger.jar:opinionfinder.jar opin.main.RunOpinionFinder ../aclImdb/" + fn + " -r preprocessor,cluefinder,subjclass")
