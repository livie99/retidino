import os   
import cv2
import numpy as np


def contour (dir, name):
    # Load the image
    image = cv2.imread(os.path.join(dir, name))

    # Check if the image was loaded successfully
    if image is not None:
        image = cv2.resize(image, (4096, 4096))
        output = image.copy()
        # Convert the image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply histogram equalization to the L channel
        l_equalized = cv2.equalizeHist(l)

        # Merge the equalized L channel with the original A and B channels
        lab_equalized = cv2.merge((l_equalized, a, b))

        # Convert the LAB image back to BGR color space
        equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the image
        detected_circles = cv2.HoughCircles(gray,
                            cv2.HOUGH_GRADIENT,
                            minDist=500,
                            dp=1.1,
                            param1=500,
                            param2=17,
                            minRadius=800,
                            maxRadius=1200)
        
        # Create a mask of the circle
        mask = np.zeros_like(gray)
        cropped_images = []
        if detected_circles is not None:
            for (x, y, r) in detected_circles[0, :, :]:
                if x+y < 2000:
                    continue
                
                # Draw the circle on the original image
                cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 30)
                
                # Draw the circle on the mask
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                
                # Create a colored mask
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Crop the region of the circle
                crop_image = masked_image[int(y-r if y-r>0 else 0):int(y+r if y+r<4096 else 4096), int(x-r if x-r>0 else 0):int(x+r if x+r<4096 else 4096)]
                crop_image = cv2.resize(crop_image, (384, 384))
                # Append the cropped image to a list
                cropped_images.append(crop_image)
                mask = np.zeros_like(mask)
            
            # print("Image processed successfully.", flush=False)
            return cropped_images
        
        else:
            # print("No circle detected. Consider change the parameters.", flush=False)
            return None
    
    else:
        print("Failed to load the image.")

def contour_check(dir, name):
    ## check why there are more than 10 circles detected in some images
    imgs = contour(dir,name)

    for i,img in enumerate(imgs):
        crop_path = f"/home/livieymli/retidino/ignore/cropped_{name}_{i}.jpg"
        cv2.imwrite(crop_path, img)

    return None 

def main():
    root_folder = "/home/livieymli/t1riskengine/data/mosaics_graded"
    for label in range(0,5):
        eg_folder = os.path.join(root_folder, str(label))
        img_names = os.listdir(eg_folder)
        img_names.sort()
        for _,name in enumerate(img_names):
            images = contour(eg_folder, name)
            # name = name.split(".")[0]
            if images is not None:
                # Save the cropped image
                # for i,img in enumerate(images):
                #     crop_path = f"/home/livieymli/retidino/ignore/cropped_{name}_{i}.jpg"
                #     cv2.imwrite(crop_path, img)
                print(f"{label}, {name}, {len(images)}", flush=True)
            else:
                print(f"{label}, {name}, -1", flush=True)
                # pass

if __name__ == "__main__":
    main()