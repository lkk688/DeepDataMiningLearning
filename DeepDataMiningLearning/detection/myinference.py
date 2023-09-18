import torch
#https://debuggercafe.com/anchor-free-object-detection-inference-using-fcos-fully-connected-one-stage-object-detection/

import cv2
import torch
import argparse
import time
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

def get_torchvision_detection_models(modelname, box_score_thresh=0.9):
    weights_enum = get_model_weights(modelname) #<enum 'FasterRCNN_MobileNet_V3_Large_320_FPN_Weights'>
    weights = weights_enum.DEFAULT #get the default weights
    preprocess = weights.transforms()
    classes = weights.meta["categories"]
    pretrained_model=get_model(modelname, box_score_thresh=0.9, weights="DEFAULT")
    return pretrained_model, preprocess, weights, classes

def test_inference(modelname, imgpath):
    img = read_image(imgpath)
    pretrained_model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
    pretrained_model.eval()
    #Apply inference preprocessing transforms
    batch = [preprocess(img)]
    prediction = pretrained_model(batch)[0]
    labels = [classes[i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    return im

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/video_1.mp4', 
                    help='path to input video')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')
args = vars(parser.parse_args())

if __name__ == "__main__":
    main(args)

def main(args):
    # Define the computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    cap = cv2.VideoCapture(args['input'])
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"
    # Define codec and create VideoWriter object .
    out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # Read until end of video.
    while(cap.isOpened):
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                boxes, classes, labels = detect_utils.predict(
                    frame, model, 
                    device, args['threshold']
                )
            
            # Draw boxes and show current frame on screen.
            image = detect_utils.draw_boxes(boxes, classes, labels, frame)
            # Get the end time.
            end_time = time.time()
            # Get the fps.
            fps = 1 / (end_time - start_time)
            # Add fps to total fps.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            # Write the FPS on the current frame.
            cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # Convert from BGR to RGB color format.
            cv2.imshow('image', image)
            out.write(image)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # Transform the image to tensor.
    image = transform(image).to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0) 
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image) 
    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # Get all the predicited class names.
    pred_classes = [coco_names[i] for i in labels.cpu().numpy()]
    return boxes, pred_classes, labels

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color[::-1], 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[::-1], 2, 
                    lineType=cv2.LINE_AA)
    return image