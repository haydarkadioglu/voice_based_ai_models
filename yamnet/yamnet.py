# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""

import csv
import numpy as np
import tensorflow as tf
from keras import layers, Model

def _batch_norm(name, params):  # Changed from create_batch_norm
    return layers.BatchNormalization(
        name=name,
        center=params.batchnorm_center,
        scale=params.batchnorm_scale,
        epsilon=params.batchnorm_epsilon
    )

def _conv_block(name, kernel, stride, filters, params):  # Changed from create_conv_block
    def conv_block(inputs):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding=params.conv_padding,
            use_bias=False,
            name=f'conv2d_{name}'
        )(inputs)
        x = _batch_norm(f'batch_normalization_{name}', params)(x)
        x = layers.ReLU(name=f'activation_{name}')(x)
        return x
    return conv_block

def _separable_conv_block(name, kernel, stride, filters, params):  # Changed from create_separable_conv_block
    def sep_conv_block(inputs):
        x = layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding=params.conv_padding,
            use_bias=False,
            name=f'separable_conv2d_{name}'
        )(inputs)
        x = _batch_norm(f'batch_normalization_{name}', params)(x)
        x = layers.ReLU(name=f'activation_{name}')(x)
        return x
    return sep_conv_block

YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv_block,          [3, 3], 2,   32),
    (_separable_conv_block, [3, 3], 1,   64),
    (_separable_conv_block, [3, 3], 2,  128),
    (_separable_conv_block, [3, 3], 1,  128),
    (_separable_conv_block, [3, 3], 2,  256),
    (_separable_conv_block, [3, 3], 1,  256),
    (_separable_conv_block, [3, 3], 2,  512),
    (_separable_conv_block, [3, 3], 1,  512),
    (_separable_conv_block, [3, 3], 1,  512),
    (_separable_conv_block, [3, 3], 1,  512),
    (_separable_conv_block, [3, 3], 1,  512),
    (_separable_conv_block, [3, 3], 1,  512),
    (_separable_conv_block, [3, 3], 2, 1024),
    (_separable_conv_block, [3, 3], 1, 1024)
]

def create_yamnet_model(params):
    """Creates the YAMNet model using Keras functional API"""
    inputs = layers.Input(shape=(params.patch_frames, params.patch_bands), name='input_features')
    
    # Reshape for convolution
    x = layers.Reshape((params.patch_frames, params.patch_bands, 1))(inputs)
    
    # Add all convolutional layers
    for i, (layer_fun, kernel, stride, filters) in enumerate(YAMNET_LAYER_DEFS):
        x = layer_fun(f'layer{i + 1}', kernel, stride, filters, params)(x)
    
    # Global pooling and final layers
    embeddings = layers.GlobalAveragePooling2D(name='embeddings')(x)
    logits = layers.Dense(units=params.num_classes, use_bias=True, name='logits')(embeddings)
    predictions = layers.Activation(params.classifier_activation, name='predictions')(logits)
    
    # Create model
    model = Model(inputs=inputs, outputs=[predictions, embeddings], name='yamnet')
    return model

def create_yamnet_frames_model(params):
    """Creates the YAMNet waveform-to-class-scores model"""
    waveform_input = layers.Input(shape=(None,), dtype=tf.float32, name='waveform')
    spectrogram = create_mel_spectrogram_layer(params)(waveform_input)
    features = extract_patches(params)(spectrogram)
    
    # Get base YAMNet model
    yamnet = create_yamnet_model(params)
    predictions, embeddings = yamnet(features)
    
    # Create the full model
    model = Model(
        inputs=waveform_input,
        outputs=[predictions, embeddings, spectrogram],
        name='yamnet_frames'
    )
    return model

def create_mel_spectrogram_layer(params):
    """Creates a custom layer for mel spectrogram computation"""
    # Implement mel spectrogram computation using Keras layers
    # This is a placeholder - you'll need to implement the actual computation
    return layers.Lambda(lambda x: x, name='mel_spectrogram')

def extract_patches(params):
    """Creates a custom layer for extracting patches"""
    # Implement patch extraction using Keras layers
    # This is a placeholder - you'll need to implement the actual computation
    return layers.Lambda(lambda x: x, name='extract_patches')

def load_class_names(class_map_csv):
    """Read the class names from CSV file"""
    if tf.is_tensor(class_map_csv):
        class_map_csv = class_map_csv.numpy()
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        return np.array([display_name for (_, _, display_name) in reader])
