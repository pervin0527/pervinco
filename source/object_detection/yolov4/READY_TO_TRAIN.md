[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

## xml -> txt

    cd pervinco/source/augmentations
    python3 xml2txt.py

## yolov4-custom.cfg

- batch = 64
- subdivisions = 16
- max_batches = classes * 2000
- steps = 80 ~ 90% of max_batches
- width, height = multiple of 32
- classes = each of 3 [yolo]-layers
- filters = (classes + 5) * 3 before each [yolo] layer

## train

    ./darknet detector train ./custom/SPC/data/spc.cata ./custom/SPC/deploy/yolov4.cfg ./custom/SPC/deploy/yolov4.conv.137