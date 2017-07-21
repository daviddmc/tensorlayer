import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from numpy.lib.arraypad import _validate_lengths

#get error metrics, for psnr, ssimr, rmse, score_ismrm
def getErrorMetrics(im_pred, im_gt, mask = None,
                    metrics_used = ['rmse', 'psnr', 'ssim']):
    if mask is not None:
        im_pred *= mask
        im_gt *= mask
    
    # flatten array
    im_pred = np.array(im_pred).astype(np.float).flatten()
    im_gt = np.array(im_gt).astype(np.float).flatten()
    #if mask is not None:
    #    mask = np.array(mask).astype(np.float).flatten()
    #    im_pred = im_pred[mask>0]
    #    im_gt = im_gt[mask>0]
    # check dimension
    assert(im_pred.flatten().shape==im_gt.flatten().shape)
    
    result = {}
    if 'mse' in metrics_used:
        result.update({'mse' : compare(im_gt. im_pred)})
    if 'rmse' in metrics_used:
        result.update({'rmse' : compare_nrmse(im_gt, im_pred)})
    if 'psnr' in metrics_used:
        result.update({'psnr' : compare_psnr(im_gt, im_pred)})
    if 'ssim' in metrics_used:
        result.update({'ssim' : compare_ssim(im_gt, im_pred)})
    if 'score_ismrm' in metrics_used:
        result.update({'score_ismrm', compare_score_ismrm(im_gt, im_pred)})
    return result


def compare_score_ismrm(im_gt, im_pred):
    mask = np.abs(im_gt.flatten())>0
    score = sum((np.abs(im_gt.flatten()-im_pred.flatten())<0.1)*mask)/(sum(mask)+1e-9)*10000
    return score
    
def compare_mse(im1, im2):
    """calculate the mean squared error of two images"""
    return np.mean(np.square(im1 - im2), dtype=np.float64)

def compare_nrmse(im_true, im_test, norm_type='Euclidean'):
    """calculate the normalized root-mean-square error
    i.e. nrmse = mse / norm_factor
    
    type of normalization:
    1. euclidean : std of the true image
    2. min-max : max(true image) - min(true image)
    3. mean : mean of the true image"""
    
    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((im_true*im_true), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = im_true.max() - im_true.min()
    elif norm_type == 'mean':
        denom = im_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(compare_mse(im_true, im_test)) / denom

def compare_psnr(im_true, im_test, pmax = 1):
    """calcualte the peak signal-to-noise ratio
    PSNR = 10 * lg(MAX^2 / mse)"""
   
    err = compare_mse(im_true, im_test)
    return 10 * np.log10((pmax ** 2) / err)

def compare_ssim(X, Y, win_size=None, gradient=False,
                 data_range=1, multichannel=False, gaussian_weights=False,
                 full=False, **kwargs):

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = compare_ssim(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')
    '''
    if data_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin
    '''
    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim

def crop(ar, crop_width, copy=False, order='K'):
    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped
