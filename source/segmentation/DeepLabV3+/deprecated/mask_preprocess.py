"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import cv2
import glob
import os.path
import numpy as np

from PIL import Image

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('original_gt_folder', '/data/Datasets/VOCdevkit/VOC2012/SegmentationClass', 'Original ground truth annotations.')

tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.compat.v1.flags.DEFINE_string('output_dir', '/data/Datasets/VOCdevkit/VOC2012/SegmentationRaw', 'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.

  Returns:
    Annotation without color map.
  """
  mask = np.array(Image.open(filename))
  seg_mask = np.where(mask == 255, 0, mask)
  print(seg_mask.shape)
  return seg_mask


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.io.gfile.GFile(filename, mode='w') as f:
    pil_image.save(f, 'PNG')


def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.io.gfile.isdir(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                       '*.' + FLAGS.segmentation_format))
  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    # image = cv2.imread(annotation)
    # cv2.imshow('image', image)
    # cv2.imshow('mask', raw_annotation)
    # cv2.waitKey(0)

    # break
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(
                         FLAGS.output_dir,
                         filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
  tf.compat.v1.app.run()