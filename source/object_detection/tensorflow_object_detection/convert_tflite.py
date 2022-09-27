import os
import tensorflow as tf
from glob import glob
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)

def preprocess_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [320, 320])
    image = image / 127.5 - 1
    # image = image / 255.

    return image

def representative_data_gen():
    images = sorted(glob("/data/Datasets/COCO2017/images/*"))
    idx = 0
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocess_image).batch(1).take(100):
        idx += 1
        
        yield [input_value]

if __name__ == "__main__":
    graph_path = '/home/jun/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    saved_model_dir = f"{graph_path}/saved_model"
    label_file_paths=f'{graph_path}/tflite_label_map.txt'

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.experimental_new_quantizer = True
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f'{graph_path}/custom.tflite', 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(f'{graph_path}/custom.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    image = tf.image.decode_image(open("./dog.jpg", "rb").read(), channels=3)
    image = tf.image.resize(image, (320, 320))
    input_tensor = tf.expand_dims(image, 0)

    interpreter.set_tensor(input_details[0]["index"], tf.cast(input_tensor, tf.uint8))
    interpreter.invoke()

    output0 = interpreter.get_tensor(output_details[0]["index"]) # scores
    output1 = interpreter.get_tensor(output_details[1]["index"]) # boxes
    output2 = interpreter.get_tensor(output_details[2]["index"])
    output3 = interpreter.get_tensor(output_details[3]["index"]) # class_ids
    print(output0.shape, output1.shape, output2.shape, output3.shape)
    print(output0[:5])
    print(output3[:5])

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "custom object detection"
    model_meta.description = ("custom object detection model for mobile tflite")
    export_model_path = f'{graph_path}/custom_metadata.tflite'

    tf.io.gfile.copy(f"{graph_path}/custom.tflite", export_model_path, overwrite=True)

    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "Image"
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = (_metadata_fb.ColorSpaceType.RGB)
    input_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.ImageProperties)
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5] # 127.0
    input_normalization.options.std = [127.5] # 128.0
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [1.0] # 255.0
    input_stats.min = [-1.0] # 0.0
    input_meta.stats = input_stats

    # Creates outputs info.
    output_location_meta = _metadata_fb.TensorMetadataT()
    output_location_meta.name = "BondingBox"
    output_location_meta.description = "The locations of the detected boxes."
    output_location_meta.content = _metadata_fb.ContentT()
    output_location_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.BoundingBoxProperties)
    output_location_meta.content.contentProperties = (_metadata_fb.BoundingBoxPropertiesT())
    output_location_meta.content.contentProperties.index = [1, 0, 3, 2]
    output_location_meta.content.contentProperties.type = (_metadata_fb.BoundingBoxType.BOUNDARIES)
    output_location_meta.content.contentProperties.coordinateType = (_metadata_fb.CoordinateType.RATIO)
    output_location_meta.content.range = _metadata_fb.ValueRangeT()
    output_location_meta.content.range.min = 2
    output_location_meta.content.range.max = 2

    output_class_meta = _metadata_fb.TensorMetadataT()
    output_class_meta.name = "Feature"
    output_class_meta.description = "The categories of the detected boxes."
    output_class_meta.content = _metadata_fb.ContentT()
    output_class_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_class_meta.content.contentProperties = (_metadata_fb.FeaturePropertiesT())
    output_class_meta.content.range = _metadata_fb.ValueRangeT()
    output_class_meta.content.range.min = 2
    output_class_meta.content.range.max = 2
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = os.path.basename(label_file_paths)
    label_file.description = "Label of objects that this model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS
    output_class_meta.associatedFiles = [label_file]

    output_score_meta = _metadata_fb.TensorMetadataT()
    output_score_meta.name = "Feature"
    output_score_meta.description = "The scores of the detected boxes."
    output_score_meta.content = _metadata_fb.ContentT()
    output_score_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.FeatureProperties)
    output_score_meta.content.contentProperties = (_metadata_fb.FeaturePropertiesT())
    output_score_meta.content.range = _metadata_fb.ValueRangeT()
    output_score_meta.content.range.min = 2
    output_score_meta.content.range.max = 2

    output_number_meta = _metadata_fb.TensorMetadataT()
    output_number_meta.name = "Feature"
    output_number_meta.description = "The number of the detected boxes."
    output_number_meta.content = _metadata_fb.ContentT()
    output_number_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
    output_number_meta.content.contentProperties = (_metadata_fb.FeaturePropertiesT())

    # Creates subgraph info.
    group = _metadata_fb.TensorGroupT()
    group.name = "detection result"
    group.tensorNames = [output_location_meta.name, output_class_meta.name, output_score_meta.name]
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_location_meta, output_class_meta, output_score_meta, output_number_meta]
    subgraph.outputTensorGroups = [group]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # metadata and the label file are written into the TFLite file
    populator = _metadata.MetadataPopulator.with_model_file(export_model_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([label_file_paths])
    populator.populate()

    # Verify the populated metadata and associated files.
    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    print("Metadata populated:")
    print(displayer.get_metadata_json())
    print("Associated file(s) populated:")
    print(displayer.get_packed_associated_file_list())

    print(f'Success!\nMetadata and the label file have been written into {export_model_path}.')