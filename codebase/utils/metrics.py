import numpy as np
import pandas as pd
import math
from scipy.stats.stats import pearsonr, spearmanr
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
# https://github.com/jterrace/pyssim/blob/master/ssim/ssimlib.py
from ssim.ssimlib import *


def per_chan_mse(predicted, target):
    """ Per-channel mean squared error. Returns a vector.
    predicted: Numpy array, dimensions have to be batch_size x num_proteins
    target: Numpy array, dimensions have to be batch_size x num_proteins
    """
    # sum squared error over batch dimension, then divide by batch size to obtain MSE per channel
    loss = np.sum(np.square(predicted - target), axis=0) / predicted.shape[0]

    return loss


def per_chan_correlation(predicted, target):
    """ Per-channel Pearson correlation. Returns a vector.
    predicted: Numpy array, dimensions have to be batch_size x num_proteins
    target: Numpy array, dimensions have to be batch_size x num_proteins
    return: Pearson's correlation coefficient
    """
    result = np.zeros(predicted.shape[1])
    for chan_idx in range(predicted.shape[1]):
        # pearsonr also returns p-values, only care about correlation here, so we use [0]
        result[chan_idx] = pearsonr(predicted[:, chan_idx], target[:, chan_idx])[0]

    return result


def histogram_mutual_information(im1, im2, num_bins=100):
    hgram, x_edges, y_edges = np.histogram2d(im1.ravel(), im2.ravel(), bins=num_bins)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# forked from https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def get_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# forked from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    empty_score :  if no positive entries identified in any of the images, return this value
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


# adapted from dice score calculation (from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137)
def overlap_perc(im1, im2, empty_score=100):
    """
    Computes the percentage of overlap, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    empty_score :  if no positive entries identified in any of the images, return this value
    Returns
    -------
    overlap : float
        Percentage overlap as a float on range [0,100].
        Maximum similarity = 100
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        print('Both images are empty.')
        return empty_score

    # overlap
    intersection = np.logical_and(im1, im2)

    return 100 * intersection.sum() / im_sum


def prep_im_for_ssim(im, scale_01=True):
    """ Prepare image for (CW-)SSIM calculation
    :param im: image with one channel (2d numpy array)
    :param scale_01: bool, whether to scale to [0,1] range
    :return: Image object with values in [0,255], in grey scale
    """
    if scale_01:
        # scale values to [0,1] interval
        scaler = MinMaxScaler()
        scaler.fit(im)
        im = scaler.transform(im)
    # map value to [0,255] domain and create PIL object
    # note: since we work one single channel images, explicit conersion to greyscale is not needed
    im = Image.fromarray(np.uint8(im * 255)) #.convert('L')
    return im


def get_cwssim(im_target, im_predicted, conv_width=30):
    """ Calculate CW-SSIM between 1-channel images
    :param im_target: Numpy array, with 1 channel, output from prep_im_for_ssim (reference image)
    :param im_predicted: Numpy array, with 1 channel, output from prep_im_for_ssim (predicted image)
    :param conv_width: width of the convolution
    :return: CW-SSIM value
    """
    ssim_obj = SSIM(im_target, size=None)
    cwssim_val = ssim_obj.cw_ssim_value(im_predicted, width=conv_width)

    return cwssim_val


def get_ssim(im_target, im_predicted, gaussian_kernel_sigma=1, gaussian_kernel_width=75):
    """ Calculate SSIM between 1-channel images
    :param im_target: Numpy array, with 1 channel, output from prep_im_for_ssim (reference image)
    :param im_predicted: Numpy array, with 1 channel, output from prep_im_for_ssim (predicted image)
    :param gaussian_kernel_sigma: sigma of gaussian kernel
    :param gaussian_kernel_width: width of gaussian kernel
    :return: SSIM value
    """
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    ssim_obj = SSIM(im_target, gaussian_kernel_1d, size=None)
    ssim_val = ssim_obj.ssim_value(im_predicted)

    return ssim_val

def get_density_bins(desired_resolution_px=32, bin_lim=None, axmax=None):
    ''' Function to get bins for density estimation
    desired_resolution_px: desired resolution in px to compute density; n_bins=1000//densitycorr_px
    bin_lim: limit of bins reach (span of linspace)
    axmax: max val to divide by desired resolution and get n_bins {1000, 1024}
    '''
    if axmax is None:
        axmax = [bin_lim + i for i in range(100) if ((bin_lim+i) % desired_resolution_px)==0][0]
    n_bins = axmax//desired_resolution_px
    x_bins = np.linspace(0, bin_lim, n_bins+1)
    y_bins = np.linspace(0, bin_lim, n_bins+1)
    
    return x_bins, y_bins
    

def density_corr(img_gt_bin, img_pred_bin, metric='pcorr', desired_resolution_px=32, bin_lim=None, axmax=None, empty_score=1):
    ''' Compute correlation of densities of a binarized signal
    img_gt_bin, img_pred_bin: binarized (0,1) images (squared - x and y of the same size)
    metric: correlation coefficient to compute {pcorr, spcorr}
    desired_resolution_px: desired resolution in px to compute density; n_bins=1000//densitycorr_px
    bin_lim: limit of bins reach (span of linspace), if None, then computed based on min img size
    axmax: max val to divide by desired resolution and get n_bins {1000, 1024}
    empty_score: score assigned if both images are empty
    '''
    if bin_lim is None:
        bin_lim = min(img_gt_bin.shape[0], img_pred_bin.shape[0])
    x_bins, y_bins = get_density_bins(desired_resolution_px, bin_lim, axmax)

    im_sum = img_gt_bin.sum() + img_pred_bin.sum()
    if im_sum == 0:
        print('Both images are empty')
        return empty_score
    else:
        # get location of non-zero entries
        coords_gt = pd.DataFrame(np.nonzero(img_gt_bin), index=['X', 'Y']).transpose()
        coords_pred = pd.DataFrame(np.nonzero(img_pred_bin), index=['X', 'Y']).transpose()
        if (coords_gt.shape[0]<1 or coords_pred.shape[0]<1):
            print('One of the images is empty, returning nan')
            return np.nan
        else:
            # compute density at desired resolution
            density_gt, _, _ = np.histogram2d(coords_gt['X'], coords_gt['Y'], [x_bins, y_bins], density=True)   
            density_pred, _, _ = np.histogram2d(coords_pred['X'], coords_pred['Y'], [x_bins, y_bins], density=True)
            # correlation between GT and Pred density
            if metric == 'pcorr':
                density_corr = pearsonr(density_gt.flatten(), density_pred.flatten())[0]
            elif metric == 'spcorr':
                density_corr = spearmanr(density_gt.flatten(), density_pred.flatten())[0]
            return density_corr