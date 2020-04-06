from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from tqdm import tqdm
import random
import re
from tensorflow.python.platform import gfile
from PIL import Image
from sklearn import preprocessing
from tqdm import tqdm

csv_path="ssTrain.csv"

def get_train_and_test(seed,csv_path):
    f_in=open(csv_path,"r")
    lines=f_in.readlines()
    random.seed(seed)
    random.shuffle(lines)
    n_train=int(0.9*len(lines))
    f_in.close()
    return lines[:n_train],lines[n_train:]

def get_paths_and_indexs(data_dir):
    paths=[]
    indexs={}
    f_all_train=open("allTrain.csv","r")
    lines=f_all_train.readlines()
    idx=0
    for line in tqdm(lines):
        member_path=data_dir+"/"+line[:-1]
        for img_name in os.listdir(member_path):
            img_path=member_path+img_name
            paths.append(img_path)
            key=line[:-1]+img_name
            indexs[key]=idx
            idx+=1
    return paths,indexs

def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_data(paths_batch, image_size):
    images=[]
    for path in paths_batch:
        image=Image.open(path).convert("RGB").resize((image_size,image_size))
        image=np.array(image)
        images.append(image)
    images=np.array(images)
    return images

def get_pairs_and_labels(dataset,indexs,emb_array,data_dir):
    emb_array1=[]
    emb_array2=[]
    labels=[]

    for line in tqdm(dataset):
        line=line[:-1]
        line=line.split(",")
        member1_path=data_dir+"/"+line[0]
        for img1_name in os.listdir(member1_path):
            member2_path=data_dir+"/"+line[1]
            for img2_name in os.listdir(member2_path):
                key1=line[0]+img1_name
                key2=line[1]+img2_name
                emb_array1.append(emb_array[indexs[key1]])
                emb_array2.append(emb_array[indexs[key2]])
                label=int(line[2])
                labels.append(label)

    return np.array(emb_array1),np.array(emb_array2),np.array(labels)

def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            train_set, test_set = get_train_and_test(args.seed,args.csv_path)
            if (args.mode == 'TRAIN'):
                dataset = train_set
            elif (args.mode == 'CLASSIFY'):
                dataset = test_set

            print("Getting paths and indexs")
            paths, indexs = get_paths_and_indexs(args.data_dir)

            print('Loading feature extraction model')
            load_model(args.model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in tqdm(range(nrof_batches_per_epoch)):

                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = load_data(paths_batch, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode == 'TRAIN'):
                print("Getting pairs and labels")
                emb_array1,emb_array2,labels=get_pairs_and_labels(dataset,indexs,emb_array,args.data_dir)
                X=np.fabs(emb_array1-emb_array2)
                Y=labels
                scaler = preprocessing.StandardScaler().fit(X)
                X=scaler.transform(X)

                model = SVC(kernel='rbf', probability=True)
                print('Training classifier')
                model.fit(X, Y)

                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, scaler), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode == 'CLASSIFY'):

                with open(classifier_filename_exp, 'rb') as infile:
                    (model, scaler) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                emb_array1,emb_array2,labels=get_pairs_and_labels(dataset,indexs,emb_array,args.data_dir)
                X=np.fabs(emb_array1-emb_array2)
                Y=labels
                X=scaler.transform(X)

                print('Testing classifier')
                predictions = model.predict_proba(X)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, best_class_indices[i], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
            help='Path to the data directory containing aligned LFW face patches.',default="./fiw")
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default="20180408-102900")
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.',default="ssClassfier.pkl")
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument("--csv_path",type=str,help="the path of csv file",default="bbTrain.csv")

    return parser.parse_args(argv)


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    main(parse_arguments(sys.argv[1:]))
