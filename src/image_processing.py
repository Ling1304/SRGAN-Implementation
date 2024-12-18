import os
import cv2

def get_lr_images(dataset_path, output_path, upscale_factor=4):
  '''
  Function to get low resolution images
  - Downscale + Upscale with Bicubic
  '''
  # Ensure output directories exist
  os.makedirs(output_path, exist_ok=True)

  # Get a list of image files from image folder path
  image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

  val_image_count = 0

  # For each image in image_file
  for image_file in image_files:
    # Read each image
    image = cv2.imread(image_file)

    # Get the height and width of image
    height, width, _ = image.shape

    # - Downscale by upscaling factor
    downscale_size = (width//upscale_factor, height//upscale_factor)
    downscale_image = cv2.resize(image, downscale_size, interpolation=cv2.INTER_LINEAR)

    # - Upscaling back to original size using bicubic interpolation
    upscale_image = cv2.resize(downscale_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Save the LR image
    original_name = os.path.basename(image_file)
    base_name, _ = os.path.splitext(original_name)
    image_name = f"lr_{base_name}.png"
    image_path = os.path.join(output_path, image_name)
    cv2.imwrite(image_path, upscale_image)

    val_image_count += 1

dataset_path = "C:/Users/hezro/Desktop/Set5+14"
lr_output_path = "C:/Users/hezro/Desktop/SRGAN-Implementation/data/train"

get_lr_images(dataset_path, lr_output_path, upscale_factor=4)
print('Done!')