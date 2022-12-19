import os
import csv
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
import numpy as np
from six import BytesIO
from PIL import Image

detection_model = None
labels = []

def loadmodel(modelname):
    global detection_model
    global labels
    # Get current path    
    curpath = os.getcwd()
    # file Upload
    uploadpath = os.path.join(curpath, 'models')
    modelpath =  os.path.join(uploadpath, modelname)
    csvpath =  os.path.join(uploadpath, modelname + ".csv")
    # parsing class num 
    labels = []
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(row['label'])

    num_classes = len(labels)
    pipeline_config = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    #ckpt.restore(checkpoint_path)
    ckpt.restore(modelpath).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    
def objectdetect(img_data):
    global detection_model
    global labels
    resultdic = []
    if detection_model == None:
        return resultdic
    
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    imgtensor = np.expand_dims(np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8), axis=0)
    input_tensor = tf.convert_to_tensor(imgtensor, dtype=tf.float32)

    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    preresult =  detection_model.postprocess(prediction_dict, shapes)

    classes = preresult['detection_classes'][0].numpy().astype(np.uint32)
    boxes = preresult['detection_boxes'][0].numpy()
    scores = preresult['detection_scores'][0].numpy()
    
    for i in range(boxes.shape[0]):
        if scores[i] > 0.8:
            box = tuple(boxes[i].tolist()) #ymin, xmin, ymax, xmax
            resultdic.append({
                            'Class': labels[classes[i]],
                            'Score': float(scores[i]),
                            'Rect': {
                                'x': int(box[1] * im_width), #miny
                                'y': int(box[0] * im_height),
                                'w': int((box[3] - box[1])  * im_width),
                                'h': int((box[2] - box[0])  * im_height),
                            }
                        })

    return resultdic

if __name__ == "__main__":
    loadmodel("testmodel")
    img_data = tf.io.gfile.GFile("test2.jpg", 'rb').read()
    candidate = objectdetect(img_data)