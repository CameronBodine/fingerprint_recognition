import cv2 as cv
from glob import glob
import os, sys
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize

from joblib import Parallel, delayed, cpu_count


def fingerprint_pipline(img_path):
    block_size = 16
    img_resize_x = 275
    img_resize_y = 275
    std_threshold = 0.05 #0.2

    # Open image
    input_img = cv.imread(img_path, 0)

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # Do resize
    input_img = cv.resize(input_img, (img_resize_x, img_resize_y), interpolation=cv.INTER_AREA)

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # color threshold
    # threshold_img = normalized_img
    # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
    # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, std_threshold)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    # minutias
    minutias = calculate_minutiaes(thin_image)

    # singularities
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # visualize pipeline stage by stage
    # output_imgs = [input_img, normalized_img, segmented_img, orientation_img, 
    #                gabor_img, thin_image, minutias, singularities_img]
    output_imgs = [input_img, normalized_img, segmented_img,  
                   orientation_img, gabor_img, thin_image]
    split_plt = int(len(output_imgs) / 2)
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate([np.concatenate(output_imgs[:split_plt], 1), np.concatenate(output_imgs[split_plt:], 1)]).astype(np.uint8)

    cv.imwrite(output_dir+os.path.basename(img_path), results)

    # sys.exit()

    # return results


if __name__ == '__main__':
    # open images
    img_dir = r'./onr_muri_proposal_in/*'
    output_dir = r'./onr_muri_proposal_out/'

    images = glob(img_dir)

    os.makedirs(output_dir, exist_ok=True)

    threadCnt = 0.75

    ###############################################
    # Specify multithreaded processing thread count
    if threadCnt==0: # Use all threads
        threadCnt=cpu_count()
    elif threadCnt<0: # Use all threads except threadCnt; i.e., (cpu_count + (-threadCnt))
        threadCnt=cpu_count()+threadCnt
        if threadCnt<0: # Make sure not negative
            threadCnt=1
    elif threadCnt<1: # Use proportion of available threads
        threadCnt = int(cpu_count()*threadCnt)
        # Make even number
        if threadCnt % 2 == 1:
            threadCnt -= 1
    else: # Use specified threadCnt if positive
        pass

    if threadCnt>cpu_count(): # If more than total avail. threads, make cpu_count()
        threadCnt=cpu_count();
        print("\nWARNING: Specified more process threads then available, \nusing {} threads instead.".format(threadCnt))

    Parallel(n_jobs=threadCnt)(delayed(fingerprint_pipline)(img_path) for img_path in tqdm(images))


