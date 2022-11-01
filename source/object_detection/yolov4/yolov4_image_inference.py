"""
./darknet detector train ./custom/obj.data ./custom/yolov4.cfg ./custom/yolov4.conv.137 -dont_show
To calculate anchors: ./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
"""

import cv2
import darknet


if __name__ == "__main__":
    WEIGHT_DIR = "/home/ubuntu/Models/BR/yolov4_last.weights"
    CONFIG_DIR = "./custom/yolov4.cfg"
    DATA_DIR = "./custom/obj.data"
    
    network, class_names, class_colors = darknet.load_network(CONFIG_DIR, DATA_DIR, WEIGHT_DIR, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread("./custom/0026.jpg")
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (width, height))
    copy_resized = resized_img.copy()

    darknet.copy_image_from_bytes(darknet_image, resized_img.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.4)
    result_image = darknet.draw_boxes(detections, resized_img, class_colors)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite("predictions.jpg", result_image)