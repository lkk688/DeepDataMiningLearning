
from DeepDataMiningLearning.networkutil import downloadurls


if __name__ == '__main__':

    visdrone_urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
    # url='https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip'#'https://www.sjsu.edu'
    # url = _get_redirect_url(url, max_hops=3)
    dir="/data/cmpe249-fa23/VisDrone"
    downloadurls(dir, visdrone_urls)



           

# mytorchvisiondata='/data/cmpe249-fa23/torchvisiondata/'
# car_dataset=datasets.StanfordCars(
#     root=mytorchvisiondata,
#     download=True,
#     transform=None,
# ) #urllib.error.HTTPError: HTTP Error 404: Not Found
# print(car_dataset)