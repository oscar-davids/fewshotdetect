import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
import csv
import os
import numpy as np
from PIL import Image
import random
from six import BytesIO
from flask import flash

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, batch_size, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      preprocessed_images = tf.concat(
          [model.preprocess(image_tensor)[0]
           for image_tensor in image_tensors], axis=0)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

  return train_step_fn
      
def trainmodel(modelname):
    # Get current path
    curpath = os.getcwd()
    # file Upload
    uploadpath = os.path.join(curpath, 'traindata')
    modelpath =  os.path.join(uploadpath, modelname)
    csvpath =  os.path.join(modelpath, "label.csv")

    num_classes = 1
    trimapath = []
    labelstr = []
    labellist = []
    gt_boxes = []

    # parsing class num and object positions
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            imgpath = os.path.join(modelpath, row['image_name'])
            if os.path.exists(imgpath) == False:
                continue

            trimapath.append(imgpath)
            labelstr.append(row['label_name'])
            ty = float(row['bbox_y']) / float(row['image_height'])
            tx = float(row['bbox_x']) / float(row['image_width'])
            by = (float(row['bbox_y']) + float(row['bbox_height'])) / float(row['image_height'])
            bx = (float(row['bbox_x']) + float(row['bbox_width'])) / float(row['image_width'])
            gt_boxes.append(np.array([[ty, tx, by, bx]], dtype=np.float32))
          

    labels = list(set(labelstr))
    num_classes = len(labels)
    for lb in labelstr:
        labellist.append(labels.index(lb))
    #GPU setting
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

    #load train images
    train_images_np = []
    for imgpath in trimapath:        
        train_images_np.append(load_image_into_numpy_array(imgpath))

    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    
    for (train_image_np, gt_box_np, lb) in zip(
        train_images_np, gt_boxes, labellist):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
        train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) + lb - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))

    tf.keras.backend.clear_session()
    pipeline_config = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    checkpoint_path = 'checkpoint/ckpt-0'

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
    fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    flash('Weights restored!')
    

    tf.keras.backend.set_learning_phase(True)
    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = 4
    learning_rate = 0.01
    num_batches = 100

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)  
    train_step_fn = get_model_train_step_function(detection_model, optimizer, batch_size, to_fine_tune)

    flash('Start fine-tuning!')
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]
            
        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        
        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)
        
        if idx % 10 == 0:
            #tf.saved_model.save(detection_model, "test.pt")  # or model.save(path, save_format='tf')
            flash('batch ' + str(idx) + ' of ' + str(num_batches)
            + ', loss=' +  str(total_loss.numpy()))

    storeckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    modelpath = os.path.join(curpath, 'models', modelname)
    storeckpt.write(modelpath)

    #write label info
    labelpath = os.path.join(curpath, 'models', modelname + ".csv")
    with open(labelpath, 'w', newline='') as csvfile:
        fieldnames = ['label', 'number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)        
        writer.writeheader()
        id = 0
        for label in labels:
            writer.writerow({'label': label, 'number': id})
            id += 1

    flash('Done fine-tuning!')

if __name__ == "__main__":
    trainmodel("testmodel")    
