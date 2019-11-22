from .base import Darkdata,COCO,PascalVOC
from  .yolo import Yolo
dataset = {
  'coco': COCO,
  'pascal': PascalVOC,
  'dark': Darkdata,
}
def get_dataset(dataset_name):
  class Dataset(dataset[dataset_name],Yolo):
    pass
  return Dataset