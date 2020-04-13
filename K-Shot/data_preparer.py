import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle


from countDict import *

def get_intent_labels(task):
    """Returns an array of all unique intent labels for a given task"""
    # get file path
    path = "../data/{}/intent_label.txt".format(task)
    text_file = open(path, "r")
    lines = text_file.readlines()
    # get rid of first label "UNK" and newlines/whitespace
    intent_labels =  [x.strip() for x in lines[1:]]
    return intent_labels

def write_to_file(outfile_label, outfile_in, outfile_out, data):
    """Write to 3 seperate outfiles for a given array of data"""
    for data_points in data:
        outfile_label.write(data_points[0]+"\n")
        outfile_in.write(data_points[1]+"\n")
        outfile_out.write(data_points[2]+"\n")

# task = the task name e.g 'atis' 'snips' 'fb-alarm'
# n = percentage as integer n=1 equals 1%
def get_n_percent_from(task, n):
    # create data frames
    df_label = pd.read_csv('{}/fullTrain/label'.format(task), names=['label'])
    df_in = pd.read_csv('{}/fullTrain/seq.in'.format(task),   names=['seq.in'])
    df_out = pd.read_csv('{}/fullTrain/seq.out'.format(task), names=['seq.out'])
    # concatenate all dfs horizontally
    df = pd.concat([df_label,df_in,df_out],axis=1)
    # shuffle rows of df
    # df = shuffle(df)
    directory = "{}/{}%".format(task, n)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # declare outputfiles
    outfile_label = open('{}/{}%/label'.format(task,n),'w')
    outfile_in = open('{}/{}%/seq.in'.format(task,n),'w')
    outfile_out = open('{}/{}%/seq.out'.format(task,n),'w')

    # get unique intent_labels from whatever task
    intent_labels = get_intent_labels(task)
    # get countDictionary
    count_dict = taskToDict[task]

    # convert int value to percentage e.g 1 ==> 0.01
    percent = n / 100.0

    for intent_label in intent_labels:
        num_examples = int(np.ceil(count_dict[intent_label] * percent))
        is_label = df['label']==intent_label
        data_points = df[is_label].values[0:num_examples]
        write_to_file(outfile_label, outfile_in, outfile_out, data_points)

    outfile_label.close()
    outfile_in.close()
    outfile_out.close()

def get_k_samples_from(task, k):
    # create data frames
    df_label = pd.read_csv('{}/fullTrain/label'.format(task), names=['label'])
    df_in = pd.read_csv('{}/fullTrain/seq.in'.format(task), names=['seq.in'])
    df_out = pd.read_csv('{}/fullTrain/seq.out'.format(task), names=['seq.out'])
    # concatenate all dfs horizontally
    df = pd.concat([df_label,df_in,df_out],axis=1)
    # shuffle rows of df
    df = shuffle(df)
    directory = "{}/K{}".format(task, k)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # declare outputfiles
    outfile_label = open('{}/K{}/label'.format(task, k), 'w')
    outfile_in = open('{}/K{}/seq.in'.format(task, k), 'w')
    outfile_out = open('{}/K{}/seq.out'.format(task, k), 'w')

    # get unique intent_labels from whatever task
    intent_labels = get_intent_labels(task)
    # get countDictionary
    count_dict = taskToDict[task]


    for intent_label in intent_labels:
        # num_examples = int(np.ceil(count_dict[intent_label] * k))
        print(intent_label)
        num_examples = k
        is_label = df['label']==intent_label
        data_points = df[is_label].values[0:num_examples]
        write_to_file(outfile_label, outfile_in, outfile_out, data_points)

    outfile_label.close()
    outfile_in.close()
    outfile_out.close()


get_n_percent_from('fb-alarm', 2)