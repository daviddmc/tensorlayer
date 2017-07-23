#! /usr/bin/python
# -*- coding: utf8 -*-


import matplotlib

## use this, if you got the following error:
#  _tkinter.TclError: no display name and no $DISPLAY environment variable

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
#import pandas as pd
import os
from . import prepro
from .metrics import getErrorMetrics


## Save images
import scipy.misc

def read_image(image, path=''):
    """ Read one image.

    Parameters
    -----------
    images : string, file name.
    path : string, path.
    """
    return scipy.misc.imread(os.path.join(path, image))

def read_images(img_list, path='', n_threads=10, printable=True):
    """ Returns all images in list by given path and name of each image file.

    Parameters
    -------------
    img_list : list of string, the image file names.
    path : string, image folder path.
    n_threads : int, number of thread to read image.
    printable : bool, print infomation when reading images, default is True.
    """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = prepro.threading_data(b_imgs_list, fn=read_image, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        if printable:
            print('read %d from %s' % (len(imgs), path))
    return imgs

def save_image(image, image_path=''):
    """Save one image.

    Parameters
    -----------
    images : numpy array [w, h, c]
    image_path : string.
    """
    try: # RGB
        scipy.misc.imsave(image_path, image)
    except: # Greyscale
        scipy.misc.imsave(image_path, image[:,:,0])


def save_images(images, size, image_path=''):
    """Save mutiple images into one single image.

    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : string.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img

    def imsave(images, size, path):
        return scipy.misc.imsave(path, merge(images, size))

    assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
    return imsave(images, size, image_path)

def W(W=None, second=10, saveable=True, shape=[28,28], name='mnist', fig_idx=2396512):
    """Visualize every columns of the weight matrix to a group of Greyscale img.

    Parameters
    ----------
    W : numpy.array
        The weight matrix
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    shape : a list with 2 int
        The shape of feature image, MNIST is [28, 80].
    name : a string
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.W(network.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)
    """
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images
    size = W.shape[0]
    n_units = W.shape[1]

    num_r = int(np.sqrt(n_units))  # 每行显示的个数   若25个hidden unit -> 每行显示5个
    num_c = int(np.ceil(n_units/num_r))
    count = int(1)
    for row in range(1, num_r+1):
        for col in range(1, num_c+1):
            if count > n_units:
                break
            a = fig.add_subplot(num_r, num_c, count)
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1],(28,28)), cmap='gray')
            # ------------------------------------------------------------
            feature = W[:,count-1] / np.sqrt( (W[:,count-1]**2).sum())
            # feature[feature<0.0001] = 0   # value threshold
            # if count == 1 or count == 2:
            #     print(np.mean(feature))
            # if np.std(feature) < 0.03:      # condition threshold
            #     feature = np.zeros_like(feature)
            # if np.mean(feature) < -0.015:      # condition threshold
            #     feature = np.zeros_like(feature)
            plt.imshow(np.reshape(feature ,(shape[0],shape[1])),
                    cmap='gray', interpolation="nearest")#, vmin=np.min(feature), vmax=np.max(feature))
            # plt.title(name)
            # ------------------------------------------------------------
            # plt.imshow(np.reshape(W[:,count-1] ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def frame(I=None, second=5, saveable=True, name='frame', cmap=None, fig_idx=12836):
    """Display a frame(image). Make sure OpenAI Gym render() is disable before using it.

    Parameters
    ----------
    I : numpy.array
        The image
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : a string
        A name to save the image, if saveable is True.
    cmap : None or string
        'gray' for greyscale, None for default, etc.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> env = gym.make("Pong-v0")
    >>> observation = env.reset()
    >>> tl.visualize.frame(observation)
    """
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images

    if len(I.shape) and I.shape[-1]==1:     # (10,10,1) --> (10,10)
        I = I[:,:,0]

    plt.imshow(I, cmap)
    plt.title(name)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def CNN2d(CNN=None, second=10, saveable=True, name='cnn', fig_idx=3119362):
    """Display a group of RGB or Greyscale CNN masks.

    Parameters
    ----------
    CNN : numpy.array
        The image. e.g: 64 5x5 RGB images can be (5, 5, 3, 64).
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> tl.visualize.CNN2d(network.all_params[0].eval(), second=10, saveable=True, name='cnn1_mnist', fig_idx=2012)
    """
    # print(CNN.shape)    # (5, 5, 3, 64)
    # exit()
    n_mask = CNN.shape[3]
    n_row = CNN.shape[0]
    n_col = CNN.shape[1]
    n_color = CNN.shape[2]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask/row))
    plt.ion()   # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for ir in range(1, row+1):
        for ic in range(1, col+1):
            if count > n_mask:
                break
            a = fig.add_subplot(col, row, count)
            # print(CNN[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # exit()
            # plt.imshow(
            #         np.reshape(CNN[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(
                        np.reshape(CNN[:,:,:,count-1], (n_row, n_col)),
                        cmap='gray', interpolation="nearest")
            elif n_color == 3:
                plt.imshow(
                        np.reshape(CNN[:,:,:,count-1], (n_row, n_col, n_color)),
                        cmap='gray', interpolation="nearest")
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)


def images2d(images=None, second=10, saveable=True, name='images', dtype=None,
                                                            fig_idx=3119362):
    """Display a group of RGB or Greyscale images.

    Parameters
    ----------
    images : numpy.array
        The images.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : a string
        A name to save the image, if saveable is True.
    dtype : None or numpy data type
        The data type for displaying the images.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> tl.visualize.images2d(X_train[0:100,:,:,:], second=10, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    """
    # print(images.shape)    # (50000, 32, 32, 3)
    # exit()
    if dtype:
        images = np.asarray(images, dtype=dtype)
    n_mask = images.shape[0]
    n_row = images.shape[1]
    n_col = images.shape[2]
    n_color = images.shape[3]
    row = int(np.sqrt(n_mask))
    col = int(np.ceil(n_mask/row))
    plt.ion()   # active mode
    fig = plt.figure(fig_idx)
    count = 1
    for ir in range(1, row+1):
        for ic in range(1, col+1):
            if count > n_mask:
                break
            a = fig.add_subplot(col, row, count)
            # print(images[:,:,:,count-1].shape, n_row, n_col)   # (5, 1, 32) 5 5
            # plt.imshow(
            #         np.reshape(images[count-1,:,:,:], (n_row, n_col)),
            #         cmap='gray', interpolation="nearest")     # theano
            if n_color == 1:
                plt.imshow(
                        np.reshape(images[count-1,:,:], (n_row, n_col)),
                        cmap='gray', interpolation="nearest")
                # plt.title(name)
            elif n_color == 3:
                plt.imshow(images[count-1,:,:],
                        cmap='gray', interpolation="nearest")
                # plt.title(name)
            else:
                raise Exception("Unknown n_color")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def tsne_embedding(embeddings, reverse_dictionary, plot_only=500,
                        second=5, saveable=False, name='tsne', fig_idx=9862):
    """Visualize the embeddings by using t-SNE.

    Parameters
    ----------
    embeddings : a matrix
        The images.
    reverse_dictionary : a dictionary
        id_to_word, mapping id to unique word.
    plot_only : int
        The number of examples to plot, choice the most common words.
    second : int
        The display second(s) for the image(s), if saveable is False.
    saveable : boolean
        Save or plot the figure.
    name : a string
        A name to save the image, if saveable is True.
    fig_idx : int
        matplotlib figure index.

    Examples
    --------
    >>> see 'tutorial_word2vec_basic.py'
    >>> final_embeddings = normalized_embeddings.eval()
    >>> tl.visualize.tsne_embedding(final_embeddings, labels, reverse_dictionary,
    ...                   plot_only=500, second=5, saveable=False, name='tsne')
    """
    def plot_with_labels(low_dim_embs, labels, figsize=(18, 18), second=5,
                                    saveable=True, name='tsne', fig_idx=9862):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        if saveable is False:
            plt.ion()
            plt.figure(fig_idx)
        plt.figure(figsize=figsize)  #in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i,:]
            plt.scatter(x, y)
            plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        if saveable:
            plt.savefig(name+'.pdf',format='pdf')
        else:
            plt.draw()
            plt.pause(second)

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from six.moves import xrange

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # plot_only = 500
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only,:])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, second=second, saveable=saveable, \
                                                    name=name, fig_idx=fig_idx)
    except ImportError:
        print("Please install sklearn and matplotlib to visualize embeddings.")

        
## My Visualization
def plot_with_error_metrics(im_ref, im_preds, mask = None,
    # display options
                    figsize = None, plot_row=1, fold_error=5, clim=[0,0.5], colormap = None,
    # title options
                    data_label='validation', method_labels=['Raw','NLM-Denoise','Proposed'],
    # saving options
                    path_to_save=None):
    '''plot ground truth, predictions, error maps and error metrics
    
    parameters
    ----------
    '''
    
    method_labels += [''] * (len(im_preds) - len(method_labels))
    
    # dim
    if im_ref.ndim<3:
        im_ref = im_ref[np.newaxis,:,:]
    if mask is None:
        mask = np.ones(shape = im_ref.shape, dtype=np.float64)
    if mask.ndim<3:
        mask = mask[np.newaxis,:,:]
    for i in range(len(im_preds)):
        if im_preds[i].ndim<3:    
            im_preds[i]=im_preds[i][np.newaxis,:,:]
    
    # apply mask
    if mask is not None:
        im_ref *= mask
        for i in range(len(im_preds)):
            im_preds[i] *= mask
    
    # error
    diffs = []
    results = []
    for i in range(len(im_preds)):
        im_tmp = im_preds[i]
        diff_tmp = np.abs(im_tmp[0,:,:]-im_ref[0,:,:])
        result_tmp = getErrorMetrics(im_tmp,im_ref,mask)
        diffs.append(diff_tmp)
        results.append(result_tmp)

    plt.figure(figsize=figsize)
    
    im_toshow=[]
    if plot_row==2:
        im_toshow.append(np.concatenate([im_ref[0,:,:]]+[x[0,:,:] for x in im_preds],axis=1))
        im_toshow.append(np.concatenate([mask[0,:,:]]+diffs,axis=1)*fold_error)
        im_toshow = np.concatenate(im_toshow,axis=0)
    else:
        im_toshow=[im_ref[0,:,:], mask[0,:,:]]
        for i in range(len(im_preds)):
            im_toshow+=[im_preds[i][0,:,:],diffs[i]*fold_error]
        im_toshow = np.concatenate(im_toshow, axis=1)
    title_plot = [', {0}(RMSE{1:.4f},PSNR{2:.2f}dB)'.format(method_labels[i], results[i]['rmse'], results[i]['psnr']) for i in range(len(im_preds))]
    title_plot = '{0}: Ref'.format(data_label) + ''.join(title_plot)
    
    # plot    
    if colormap is None:
        plt.imshow(im_toshow,clim=clim)
    else:
        plt.imshow(im_toshow,clim=clim,cmap=colormap)
    plt.title(title_plot)
    plt.axis('off')

    # save
    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight')   

COLOR = ['b', 'r', 'g', 'y', 'c', 'm'] 
   
def bar(mean, std=None, 
        xlabel='', ylabel='', title='', labels = None, 
        ymin=None, ymax=None):
    """make a bar plot
    
    parameters
    ----------
    mean : height of the bar
    std : error of the bar
    xlabel : label of x axis
    ylable : label of y axis
    title : title of the figure
    labels : label of each group
    ymin : minimum of y axis
    ymax : maximum of y axis
    """
    
    if labels is None:
        labels = [''] * mean.shape[1]
    assert len(labels) == mean.shape[1]    
    n_groups = mean.shape[0]  
    fig, ax = plt.subplots()  
    index = np.arange(n_groups)  
    bar_width = 0.2
    group_width = len(labels) * bar_width + 2*bar_width
    opacity = 0.4
    error_kw = dict(capsize = bar_width/2, elinewidth = 1)
    for i in range(len(labels)):
        plt.bar(index*group_width + i*bar_width, mean[:,i], bar_width, 
                alpha=opacity, color=COLOR[np.mod(i, len(COLOR))],
                label=labels[i], yerr =std[:,i], error_kw=error_kw)  
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.title(title)  
    plt.xticks(index*group_width + len(labels) * bar_width / 2.0, range(1,n_groups+1))
    if ymin is not None and ymax is not None:
        plt.ylim(ymin,ymax)  
    plt.legend()  
    plt.tight_layout()  
    plt.show() 
    
def box(data, xlabel = '', ylabel = '', title = '', labels = None, showmeans = True):
    """Make a box and whisker plot.
    
    parameters
    ----------
    data : m x n array, n is number of groups and m is number of entries in each group
    xlabel : label of x axis
    ylable : label of y axis
    title : title of the figure
    labels : label of each group
    """
    boxprops = dict(color='b')
    flierprops = dict(marker='+')
    medianprops = dict(linewidth=2, color='r') 
    meanpointprops = dict(marker='D', markeredgecolor='g',markerfacecolor='g')
   
    plt.boxplot(data, labels=labels, showmeans = showmeans,
		boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



def plot_image_zoom(imgs, layout = None,  start = (0,0), size = None, cmap=None, titles = None, mask = None):
	
	if type(imgs) is not list:
		imgs = [imgs]
	
	if size is None:
		size = (imgs[0].shape[0]//2, imgs[0].shape[1]//2)
	assert start[0] >= 0 and start[1] >= 0 and size[0] > 0 and size[1] > 0
	assert start[0] + size[0] <= imgs[0].shape[0] and start[1] + size[1] <= imgs[0].shape[1]

	if layout is None:
		layout = (1, len(imgs))
	
	if titles is None:
		titles = [''] * len(imgs)
	
	if mask is not None:
		imgs = [img * mask for img in imgs]

	plt.figure()
	
	for i in range(layout[0]):
		for j in range(layout[1]):
			idx = i*layout[1] + j
			p1 = plt.subplot(layout[0], layout[1]*2, 2*idx+1, aspect=1)
			p2 = plt.subplot(layout[0], layout[1]*2, 2*idx+2, aspect=1)

			p1.imshow(imgs[idx], cmap=cmap)
			p1.set_xticks([])
			p1.set_yticks([])
			p2.imshow(imgs[idx][start[0]:start[0]+size[0], start[1]:start[1]+size[1]], cmap=cmap)
			p2.set_xticks([])
			p2.set_yticks([])

			p1.set_title(titles[idx])

			# plot the box
			tx0 = start[1]
			tx1 = start[1] + size[1]
			ty0 = start[0]
			ty1 = start[0] + size[0]
			sx = [tx0,tx1,tx1,tx0,tx0]
			sy = [ty0,ty0,ty1,ty1,ty0]
			p1.plot(sx,sy,"b")

			# plot patch lines
			xy=(start[1] + size[1], start[0] + size[0])
			xy2 = (0,size[0]-1)
			con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
					axesA=p2,axesB=p1,color='g')
			p2.add_artist(con)

			xy = (start[1] + size[1], start[0])
			xy2 = (0,0)
			con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
					axesA=p2,axesB=p1,color='g')
			p2.add_artist(con)
	plt.tight_layout()
	plt.show()
	
    
#
