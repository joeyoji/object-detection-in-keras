import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation, Input, ZeroPadding2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.regularizers import l2
from custom_layers import L2Normalization, DefaultBoxes, DecodeSSDPredictions
from utils.ssd_utils import get_number_default_boxes

def ssd512_resnet101(
    config,
    label_maps,
    num_predictions=10,
    is_training=True,
):
    """ This network follows the official caffe implementation of SSD: https://github.com/chuanqi305/ssd
    1. Changes made to VGG16 config D layers:
        - fc6 and fc7 is converted into convolutional layers instead of fully connected layers specify in the VGG paper
        - atrous convolution is used to turn fc6 and fc7 into convolutional layers
        - pool5 size is changed from (2, 2) to (3, 3) and its strides is changed from (2, 2) to (1, 1)
        - l2 normalization is used only on the output of conv4_3 because it has different scales compared to other layers. To learn more read SSD paper section 3.1 PASCAL VOC2007
    2. In Keras:
        - padding "same" is equivalent to padding 1 in caffe
        - padding "valid" is equivalent to padding 0 (no padding) in caffe
        - Atrous Convolution is referred to as dilated convolution in Keras and can be used by specifying dilation rate in Conv2D
    3. The name of each layer in the network is renamed to match the official caffe implementation

    Args:
        - config: python dict as read from the config file
        - label_maps: A python list containing the classes
        - num_predictions: The number of predictions to produce as final output
        - is_training: whether the model is constructed for training purpose or inference purpose

    Returns:
        - A keras version of SSD300 with VGG16 as backbone network.

    Code References:
        - https://github.com/chuanqi305/ssd
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/models/keras_ssd300.py

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """
    #==========================#
    #===== config setting =====#
    #==========================#

    model_config = config["model"]
    input_shape = (model_config["input_size"], model_config["input_size"], 3)

    num_classes = len(label_maps) + 1  # for background class
    l2_reg = model_config["l2_regularization"]
    kernel_initializer = model_config["kernel_initializer"]
    default_boxes_config = model_config["default_boxes"]
    extra_default_box_for_ar_1 = default_boxes_config["extra_box_for_ar_1"]
    clip_default_boxes = default_boxes_config["clip_boxes"]

    #============================#
    #===== code contraction =====#
    #============================#

    def conv_block_1x1(filters, name, padding='same', dilation_rate=(1, 1), strides=(1, 1)):
      return Conv2D(filters, kernel_size=(1, 1), strides=strides, activation='relu', padding=padding,
                    dilation_rate=dilation_rate, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name=name)

    def conv_block_3x3(filters, name, padding='valid', dilation_rate=(1, 1), strides=(1, 1)):
      return Conv2D(filters, kernel_size=(3, 3), strides=strides, activation='relu', padding=padding,
                    dilation_rate=dilation_rate, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg), name=name)

    #============================#
    #===== resnet101 layers =====#
    #============================#

    #=== original resnet101 ===#
    resnet101 = ResNet101(include_top=True, weights="imagenet", input_tensor=Input(input_shape),
                          input_shape=None, pooling=None, classes=1000,)
    
    #=== layer from resnet101 ===#
    shape_list = []

    #(512,512,3) -> (256,256,64)
    shape_list.append((256,256,64))
    layer_level1_from_resnet101 = Model(inputs=resnet101.input,
                                        outputs=resnet101.get_layer('conv1_relu').output,
                                        name='layer_level1_from_resnet101')
    #(256,256,64) -> (128,128,256)
    shape_list.append((128,128,256))
    layer_level2_from_resnet101 = Model(inputs=resnet101.get_layer('pool1_pad').input,
                                        outputs=resnet101.get_layer('conv2_block3_out').output,
                                        name='layer_level2_from_resnet101')
    #(128,128,256) -> (64,64,512)
    shape_list.append((64,64,512))
    layer_level3_from_resnet101 = Model(inputs=resnet101.get_layer('conv3_block1_1_conv').input,
                                        outputs=resnet101.get_layer('conv3_block4_out').output,
                                        name='layer_level3_from_resnet101')
    #(64,64,512) -> (32,32,1024)
    shape_list.append((32,32,1024))
    layer_level4_from_resnet101 = Model(inputs=resnet101.get_layer('conv4_block1_1_conv').input,
                                        outputs=resnet101.get_layer('conv4_block23_out').output,
                                        name='layer_level4_from_resnet101')

    #======================#
    #===== ssd layers =====#
    #======================#

    #=== layer from ssd ===#

    # the pattern should be convX_1 padding "same", convX_2 padding "valid"
    #(32,32,1024) -> (16,16,512)
    shape_list.append((16,16,512))
    layer_level5_from_ssd512 = Sequential([
                                           Input((32,32,1024)),
                                           conv_block_1x1(256, name="conv8_1"),
                                           ZeroPadding2D(name="conv8_1_zp"),
                                           conv_block_3x3(512, strides=(2, 2), name="conv8_2"),
                                           ],name='layer_level5_from_ssd512')

    #(16,16,512) -> (8,8,256)
    shape_list.append((8,8,256))
    layer_level6_from_ssd512 = Sequential([
                                           Input((16,16,512)),
                                           conv_block_1x1(128, name="conv9_1"),
                                           ZeroPadding2D(name="conv9_1_zp"),
                                           conv_block_3x3(256, strides=(2, 2), name="conv9_2"),
                                           ],name='layer_level6_from_ssd512')

    #(8,8,256) -> (4,4,256)
    shape_list.append((4,4,256))
    layer_level7_from_ssd512 = Sequential([
                                           Input((8,8,256)),
                                           conv_block_1x1(128, name="conv10_1"),
                                           conv_block_3x3(256, name="conv10_2"),
                                           conv_block_1x1(128, name="conv11_1"),
                                           conv_block_3x3(256, name="conv11_2"),
                                           ],name='layer_level7_from_ssd512')   

    #==================#
    #===== fusion =====#
    #==================#

    #=== model construction ===#

    main3_shape = Input((64,64,512),name='layer_level3_input')
    main4_shape = Input((32,32,1024),name='layer_level4_input')
    main5_shape = Input((16,16,512),name='layer_level5_input')

    fusion_line1_from_main3 = conv_block_1x1(256,name="fusion_line1_from_main3")(main3_shape)
    fusion_line1_from_main4 = conv_block_1x1(256,name="fusion_line1_from_main4")(main4_shape)
    fusion_line1_from_main5 = conv_block_1x1(256,name="fusion_line1_from_main5")(main5_shape)

    fusion_line2_from_main4 = UpSampling2D((2,2),interpolation='bilinear',name="fusion_line2_from_main4")(fusion_line1_from_main4)
    fusion_line2_from_main5 = UpSampling2D((4,4),interpolation='bilinear',name="fusion_line2_from_main5")(fusion_line1_from_main5)

    fusion_line3 = Concatenate(name="fusion_line3")([fusion_line1_from_main3,fusion_line2_from_main4,fusion_line2_from_main5])

    fusion_line4 = conv_block_1x1(512,name="fusion_line4")(fusion_line3)

    Fusion345 = Model(inputs=[main3_shape,main4_shape,main5_shape],outputs=fusion_line4,name="fusion345")

    #=======================#
    #===== main stream =====#
    #=======================#
    
    main0 = Input(input_shape,name='layer_level0')
    main1 = layer_level1_from_resnet101(main0)
    main2 = layer_level2_from_resnet101(main1)
    main3 = layer_level3_from_resnet101(main2)
    main4 = layer_level4_from_resnet101(main3)
    main5 = layer_level5_from_ssd512(main4)
    main6 = layer_level6_from_ssd512(main5)
    main7 = layer_level7_from_ssd512(main6)
    main3 = Fusion345([main3,main4,main5]) #fusion345

    #======================================#
    #===== conf, loc, & default_boxes =====#
    #======================================#

    scales = np.linspace(
        default_boxes_config["min_scale"],
        default_boxes_config["max_scale"],
        len(default_boxes_config["layers"])
    )
    mbox_conf_layers = []
    mbox_loc_layers = []
    mbox_default_boxes_layers = []
    for i, layer in enumerate(default_boxes_config["layers"]):
        num_default_boxes = get_number_default_boxes(
            layer["aspect_ratios"],
            extra_box_for_ar_1=extra_default_box_for_ar_1
        )
        
        layer_name = layer["name"]

        layer_MBOX_CONF = Sequential([
                                      Conv2D(filters=num_default_boxes * num_classes,
                                             kernel_size=(3, 3),padding='same',
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=l2(l2_reg),
                                             name=f"{layer_name}_mbox_conf"),
                                      Reshape((-1, num_classes), name=f"{layer_name}_mbox_conf_reshape"),
                                      ],name=f'{layer_name}_MBOX_CONF')

        layer_MBOX_LOC = Sequential([
                                     Conv2D(filters=num_default_boxes * 4,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=l2(l2_reg),
                                            name=f"{layer_name}_mbox_loc"),
                                     Reshape((-1, 4), name=f"{layer_name}_mbox_loc_reshape"),
                                     ],name=f'{layer_name}_MBOX_LOC')

        layer_DEFAULT_BOXES = Sequential([
                                          DefaultBoxes(image_shape=input_shape,
                                                       scale=scales[i],
                                                       next_scale=scales[i+1] if i+1 <= len(default_boxes_config["layers"]) - 1 else 1,
                                                       aspect_ratios=layer["aspect_ratios"],
                                                       variances=default_boxes_config["variances"],
                                                       extra_box_for_ar_1=extra_default_box_for_ar_1,
                                                       clip_boxes=clip_default_boxes,name=f"{layer_name}_default_boxes"),
                                          Reshape((-1, 8), name=f"{layer_name}_default_boxes_reshape"),
                                          ],name=f'{layer_name}_DEFAULT_BOXES')

        def CONFAULOC():

          x = Input(shape_list[i+2])
          CONF = layer_MBOX_CONF(x)
          LOC = layer_MBOX_LOC(x)
          FAUL = layer_DEFAULT_BOXES(x)
          return Model(inputs=x,
                       outputs=[CONF,LOC,FAUL],
                       name=f'{layer_name}_CONFAULOC')

        layer_mbox_conf_reshape, layer_mbox_loc_reshape, layer_default_boxes_reshape = CONFAULOC()(eval(f'main{i+3}'))

        mbox_conf_layers.append(layer_mbox_conf_reshape)
        mbox_loc_layers.append(layer_mbox_loc_reshape)
        mbox_default_boxes_layers.append(layer_default_boxes_reshape)

    # concentenate class confidence predictions from different feature map layers
    mbox_conf = Concatenate(axis=-2, name="mbox_conf")(mbox_conf_layers)
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    # concentenate object location predictions from different feature map layers
    mbox_loc = Concatenate(axis=-2, name="mbox_loc")(mbox_loc_layers)
    # concentenate default boxes from different feature map layers
    mbox_default_boxes = Concatenate(axis=-2, name="mbox_default_boxes")(mbox_default_boxes_layers)
    # concatenate confidence score predictions, bounding box predictions, and default boxes
    predictions = Concatenate(axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_default_boxes])

    if is_training:
        return Model(inputs=main0,
                     outputs=predictions,
                     name='ssd512_resnet101')

    decoded_predictions = DecodeSSDPredictions(
        input_size=model_config["input_size"],
        num_predictions=num_predictions,
        name="decoded_predictions"
    )(predictions)

    return Model(inputs=main0,
                 outputs=decoded_predictions,
                 name='ssd512_resnet101')
