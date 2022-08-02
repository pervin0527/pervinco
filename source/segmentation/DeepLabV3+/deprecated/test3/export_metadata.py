import os
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

export_directory = "/data/Models/segmentation/BACKUP"
model_basename = "model_fp32_meta"
displayer = _metadata.MetadataDisplayer.with_model_file(f"{export_directory}/{model_basename}.tflite")
export_json_file = os.path.join(export_directory, os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)