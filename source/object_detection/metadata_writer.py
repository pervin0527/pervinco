from tflite_support.metadata_writers import object_detector as metadata_writer
from tflite_support.metadata_writers import writer_utils

tflite_filepath = "./yolov5s-fp16.tflite"
label_filepath = "./label_map.txt"
mean_rgb = 127.0
stddev_rgb =  128.0

writer = metadata_writer.MetadataWriter.create_for_inference(writer_utils.load_file(tflite_filepath), [mean_rgb], [stddev_rgb],  [label_filepath])
writer_utils.save_file(writer.populate(), tflite_filepath)

metadata_json = writer.get_populated_metadata_json()
with open("metadata.json", 'w') as f:
    f.write(metadata_json)