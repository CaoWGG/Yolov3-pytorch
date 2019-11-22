# yolov3-pytroch:
## Calculate the gradient value directly like Darknet.
## coco2017 val,608x608 map
| |IOU|area|maxDets| yolov3 paper| this impl|
|----------------|------------|----------|--------|---------------|----|
| Average Precision  (AP) | IoU=0.50:0.95 |   all | 100  | 0.33|0.343 |
| Average Precision  (AP) | IoU=0.50      |    all |100  | 0.579|0.572 |
| Average Precision  (AP) | IoU=0.75      |   all | 100  | 0.344|0.365 |
|Average Precision  (AP) | IoU=0.50:0.95 | small | 100  | 0.183|0.181 |
|Average Precision  (AP) | IoU=0.50:0.95 | medium | 100  |0.354| 0.377 |
| Average Precision  (AP) | IoU=0.50:0.95 | large | 100 | 0.419|0.451 |

## coco2017 val,416x416 map
| |IOU|area|maxDets| this impl|
|----------------|------------|----------|--------|---------------|
| Average Precision  (AP) | IoU=0.50:0.95 |   all | 100  |0.321 |
| Average Precision  (AP) | IoU=0.50      |    all |100 |0.541 |
| Average Precision  (AP) | IoU=0.75      |   all | 100  |0.335 |
|Average Precision  (AP) | IoU=0.50:0.95 | small | 100  |0.140 |
|Average Precision  (AP) | IoU=0.50:0.95 | medium | 100  | 0.349 |
| Average Precision  (AP) | IoU=0.50:0.95 | large | 100 |0.478 |
