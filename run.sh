#########################################################################
# File Name: run.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: 2020年03月11日 星期三 20时33分08秒
#########################################################################
#!/bin/bash
#python kinship_identify.py TRAIN --classifier_filename bbClassifier.pkl --csv_path bbTrain.csv
#python kinship_identify.py CLASSIFY --classifier_filename bbClassifier.pkl --csv_path bbTrain.csv
python kinship_identify.py TRAIN --classifier_filename fdClassifier.pkl --csv_path fdTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename fdClassifier.pkl --csv_path fdTrain.csv
python kinship_identify.py TRAIN --classifier_filename fsClassifier.pkl --csv_path fsTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename fsClassifier.pkl --csv_path fsTrain.csv
python kinship_identify.py TRAIN --classifier_filename gfgdClassifier.pkl --csv_path gfgdTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename gfgdClassifier.pkl --csv_path gfgdTrain.csv
python kinship_identify.py TRAIN --classifier_filename gfgsClassifier.pkl --csv_path gfgsTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename gfgsClassifier.pkl --csv_path gfgsTrain.csv
python kinship_identify.py TRAIN --classifier_filename gmgdClassifier.pkl --csv_path gmgdTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename gmgdClassifier.pkl --csv_path gmgdTrain.csv
python kinship_identify.py TRAIN --classifier_filename gmgsClassifier.pkl --csv_path gmgsTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename gmgsClassifier.pkl --csv_path gmgsTrain.csv
python kinship_identify.py TRAIN --classifier_filename mdClassifier.pkl --csv_path mdTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename mdClassifier.pkl --csv_path mdTrain.csv
python kinship_identify.py TRAIN --classifier_filename msClassifier.pkl --csv_path msTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename msClassifier.pkl --csv_path msTrain.csv
python kinship_identify.py TRAIN --classifier_filename sibsClassifier.pkl --csv_path sibsTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename sibsClassifier.pkl --csv_path sibsTrain.csv
python kinship_identify.py TRAIN --classifier_filename ssClassifier.pkl --csv_path ssTrain.csv
python kinship_identify.py CLASSIFY --classifier_filename ssClassifier.pkl --csv_path ssTrain.csv

