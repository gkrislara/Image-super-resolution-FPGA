import os
import argparse
import sys

import keras
from keras import backend
from keras.models import model_from_json, load_model
import tensorflow as tf

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def keras2tf(keras_json,keras_hdf5,tfckpt,tf_graph):
        
    # set learning phase for no training
    backend.set_learning_phase(0)

    if (keras_json != ''):
        json_file = open(keras_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(keras_hdf5)

    else:
        loaded_model = load_model(keras_hdf5)


    print ('Keras model information:')
    print (' Input names :',loaded_model.inputs)
    print (' Output names:',loaded_model.outputs)
    print('-------------------------------------')

    # set up tensorflow saver object
    saver = tf.train.Saver()

    # fetch the tensorflow session using the Keras backend
    tf_session = backend.get_session()

    # get the tensorflow session graph
    input_graph_def = tf_session.graph.as_graph_def()


    # get the TensorFlow graph path, flilename and file extension
    tfgraph_path = os.path.dirname(tf_graph)
    tfgraph_filename = os.path.basename(tf_graph)
    _, ext = os.path.splitext(tfgraph_filename)

    if ext == '.pbtxt':
        asText = True
    else:
        asText = False

    # write out tensorflow checkpoint & inference graph for use with freeze_graph script
    save_path = saver.save(tf_session, tfckpt)
    tf.train.write_graph(input_graph_def, tfgraph_path, tfgraph_filename, as_text=asText)

    print ('TensorFlow information:')
    print (' Checkpoint saved as:',tfckpt)
    print (' Graph saved as     :',os.path.join(tfgraph_path,tfgraph_filename))
    print('-------------------------------------')

    return


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('-kj', '--keras_json',
                    type=str,
                    default='',
    	            help='path of Keras JSON. Default is empty string to indicate no JSON file')    
    ap.add_argument('-kh', '--keras_hdf5',
                    type=str,
                    default='./model.hdf5',
    	            help='path of Keras HDF5. Default is ./model.hdf5')
    ap.add_argument('-tfc', '--tfckpt',
                    type=str,
                    default='./tfchkpt.ckpt',
    	            help='path of TensorFlow checkpoint. Default is ./tfchkpt.ckpt')
    ap.add_argument('-tfg', '--tf_graph',
                    type=str,
                    default='./tf_graph.pb',
    	            help='path of TensorFlow graph. Default is ./tf_graph.pb')
    args = ap.parse_args()


    print('\n------------------------------------')
    print('Keras version      :',keras.__version__)
    print('TensorFlow version :',tf.__version__)
    print('Python version     :',(sys.version))
    print('-------------------------------------')
    print('keras_2_tf command line arguments:')
    print(' --keras_json:', args.keras_json)
    print(' --keras_hdf5:', args.keras_hdf5)
    print(' --tfckpt    :', args.tfckpt)
    print(' --tf_graph  :', args.tf_graph)
    print('-------------------------------------')

    keras2tf(args.keras_json,args.keras_hdf5,args.tfckpt,args.tf_graph)



if __name__ == '__main__':
  main()


