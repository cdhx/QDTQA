import csv
import numpy as np


test_tsv = '/home/home2/xxhu/bert/data/test.tsv'
test_results_tsv = '/home/home2/xxhu/bert/data/test_results.tsv'


tsv_labels_test = []
tsv_data_test = []
tsv_data_result = []


with open(test_tsv, 'r') as tsv_in:
    tsv_reader = csv.reader(tsv_in, delimiter='\t')

    # read the first line that holds column labels
    tsv_labels_test = tsv_reader.__next__()

    # iterate through all the Records
    for record in tsv_reader:
        tsv_data_test.append(record)

# print(tsv_labels_test)

test_labels = [int(record[0]) for record in tsv_data_test]
# print(test_labels[0:10])


with open(test_results_tsv, 'r') as tsv_in:
    tsv_reader = csv.reader(tsv_in, delimiter='\t')
    # read the first line that holds column labels
    tsv_labels_results = tsv_reader.__next__()

    # iterate through all the Records
    for record in tsv_reader:
        tsv_data_result.append(record)


tsv_data_result = [list(float(prob) for prob in record)
                   for record in tsv_data_result]

test_labels_resut = [np.argmax(np.array(record)) for record in tsv_data_result]
# print(test_labels[0:10])


TP = 0
FP = 0
FN = 0
TN = 0

for i, label in enumerate(test_labels):
    # print(i)
    prediction = test_labels_resut[i]
    if(label == 1):
        if(prediction == 1):
            TP += 1
        else:
            FN += 1
    else:
        if(prediction == 0):
            TN += 1
        else:
            FP += 1

precision = TP/(TP+FP)
recall = TP/(TP+FN)

F1 = 2/(1/precision+1/recall)

# print(TP)
# print(FP)
# print(TN)
# print(FN)
print("P:%f\tR:%f\tF:%f" % (precision, recall, F1))
