import cv2
import os
from ultralytics import YOLO
import torchvision.transforms as T
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# video_name = r"output.mp4"
video_name = '2024-01-18T10_13_14Z-left.mp4'

# yolo_models = [(YOLO('yolov8n_hens.pt'), 0.1, 0), (YOLO('yolov8s_eggs.pt'), 0.5, 1)]
yolo_model = YOLO('best.pt')

save_result = True

# Create color_list (blue, yellow) for the masks (hens and eggs)
color_list = [(255, 0, 0), (0, 255, 255)] # Front camera

min_size = 500

roi_margins = [(100, 57), (287, 239)]
margin = 40
roi_margins = [(100-margin, 57-margin), (287+margin, 239+margin)]
roi_margins = [(0, 0), (800, 600)]

num_frame = 0
count = 0

if save_result:
    output_video = cv2.VideoWriter('./' + video_name.split('.')[0] + '_result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 9, (1280, 480))

# Open the video
video = cv2.VideoCapture(os.path.join(video_name))

# Results log (one for each model)
results_log = []

# Resuls
hens_scores = []
num_eggs = []
num_hens = []
plot_x = []

# Read the video frame by frame until the end
while video.isOpened():
    num_frame += 1
    count_miss = 0
    num_class = [0, 0]
    #if num_frame > 1:
    #    break
    
    video_ret, frame = video.read()

    # Get the current number of frame
    frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)

    # If the video is finished break
    if not video_ret:
        break

    # Taking the size of the roi_margins was set on a 400x300 image rescale to the image size
    roi_margins_scaled = [(int(roi_margins[0][0] * (frame.shape[1] / 400)), int(roi_margins[0][1] * (frame.shape[0] / 300))), (int(roi_margins[1][0] * (frame.shape[1] / 400)), int(roi_margins[1][1] * (frame.shape[0] / 300)))]

    print(roi_margins_scaled)

    # Crop the roi
    img = frame[roi_margins_scaled[0][1]:roi_margins_scaled[1][1], roi_margins_scaled[0][0]:roi_margins_scaled[1][0]]

    # Resize image to 256x256
    img = cv2.resize(img, (256, 256))

    # Increment the contrast, the brightness and the saturation of the image
    #img = cv2.convertScaleAbs(img, alpha=2.5, beta=0)

    # Create a new image with the same height as the original one and twice the width
    result_img = np.zeros((img.shape[0], img.shape[1] * 2, img.shape[2]), dtype=np.uint8)

    # If image is none skip
    if img is None:
        continue

    # Copy the image
    original_img = img.copy()

    # Save the image to temp.jpg
    cv2.imwrite('temp_test.jpg', img)

    # Iterate through all models get the index, conf and model and get the index

    # Get the index, conf and model
    try:
        results = list(yolo_model('temp_test.jpg', conf = 0.2, iou = 0.25))
        result = results[0]
    except:
        result.masks = None
    
    # If masks is none skip
    if result.masks is None:
        # Place the result image on the left side
        result_img[:, :img.shape[1], :] = original_img

        # Place the original image on the right side
        result_img[:, img.shape[1]:, :] = original_img

        # Append the results
        # results_log.append([frame_number, None, None, None])
    else:
        result = result.cpu()

        # Convert the masks matrix to a numpy array
        masks = result.masks.data.numpy()

        # Convert the boxes matrix to a numpy array
        boxes = result.boxes.xyxy.numpy()

        # Estimate the number of pixels in the boxes
        box_pixels = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])            

        # Convert class ids to a numpy array
        class_ids = result.boxes.cls.numpy()

        # Create mask image of three channels
        mask_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # Convert masks to a binary matrix
        # mask_binary = masks.astype(bool)

        # results_log.append([frame_number, boxes, class_ids, result.boxes.conf.numpy()])
        # results_log.append([frame_number, class_ids])

        # Append the results
        # results_log_object.append(current_results, index)                

        # Iterate through all the masks (first dimension of the array)
        for i in range(masks.shape[0]):
            # If the number of pixels in the box is less than min_size skip
            if box_pixels[i] < min_size:
                count_miss +=1
                continue

            # Convert current class id to integer
            current_class_id = int(class_ids[i])

            current_mask = masks[i, :, :]
            # Reshape the mask to the original image size
            current_mask = cv2.resize(current_mask, (img.shape[1], img.shape[0]))

            # Generate a random color
            color = np.random.randint(0, 255, size=(3,))

            # Apply the mask to the image as a red layer with 50% transparency
            mask_img[current_mask == 1] = color_list[current_class_id]

            current_box = boxes[i, :]
            # Convert current box to integers
            current_box = current_box.astype(int)

            # Draw the bounding box
            cv2.rectangle(img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), color_list[current_class_id], 1)

        results_log.append([frame_number, num_class])

        # Apply the mask to the img with 50% transparency
        img = cv2.addWeighted(mask_img, 0.5, img, 1, 0)

        # Place the result image on the left side
        result_img[:, :img.shape[1], :] = img

        # Place the original image on the right side
        result_img[:, img.shape[1]:, :] = original_img

    # Paste the result_img to the original frame
    # frame[roi_margins_scaled[0][1]:roi_margins_scaled[1][1], roi_margins_scaled[0][0]:roi_margins_scaled[1][0]] = img

    # Resize the image to 640x480
    img = cv2.resize(img, (640, 480))

    # Crop the roi on the original frame
    # original_roi = original_img[roi_margins_scaled[0][1]:roi_margins_scaled[1][1], roi_margins_scaled[0][0]:roi_margins_scaled[1][0]]
    original_roi = original_img.copy()

    # Resize the roi to 640x480
    original_roi = cv2.resize(original_roi, (640, 480))

    # Create a new img_result with the same height as the original one and twice the width
    img_result = np.zeros((img.shape[0], img.shape[1] * 2, img.shape[2]), dtype=np.uint8)
    # img_result = np.zeros((img.shape[0] * 2, img.shape[1] * 2, 3), np.uint8)

    # Put the img on the left side
    img_result[:, :img.shape[1], :] = img
    # Get only the red channel of the original_roi
    # original_roi = original_roi[:, :, 2]
    # Convert the original_roi to 3 channels
    # original_roi = cv2.cvtColor(original_roi, cv2.COLOR_GRAY2BGR)
    # Put the original_roi on the right side
    img_result[:, img.shape[1]:, :] = original_roi

    # Check if np.mean(hens_scores) is nan
    if np.isnan(np.mean(hens_scores)):
        # Remove all nan values
        hens_scores = [x for x in hens_scores if not np.isnan(x)]

    # Put on the top right in yellow the mean hens score
    # cv2.putText(img_result, str(round(np.mean(hens_scores), 2)) + "%", (img.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Write the frame to the output video
    # output_image = cv2.resize(img, (640, 480))
    # output_video.write(output_image)
    if save_result:
        output_video.write(img_result)
    
    # Show the image
    cv2.imshow('image', img_result)
    # cv2.waitKey(0)

    # If user press space stop the video until user press space again
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.waitKey(0)

# cv2.imshow('image', img_result)
# cv2.waitKey(0)
# Release the video
video.release()
# print(results_log)

# Release the video
if save_result:
    output_video.release()