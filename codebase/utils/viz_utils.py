import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from codebase.utils.eval_utils import *
from codebase.utils.metrics import *
# TODO: move title creation to an argument so that import from eval_utils and metrics is not needed 

def plot_binary_overlap(img_gt, img_pred, label1='GT', label2='Predicted', color1='tab:blue', color2='tab:orange', alpha=0.4, figsize=None, ax=None,
                        add_title='', add_eval=True, add_legend=True, save_fname=None, plot_show=True):
    ''' Plot overlay of binarized maps (two)
    '''
    
    colors = dict({label1:color1,label2:color2})
    cmap_gt = ListedColormap(['none', colors[label1]])
    cmap_pred = ListedColormap(['none', colors[label2]])
    cmaps = [cmap_gt, cmap_pred]
    labels = [x for x in colors.keys()]

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(img_gt, cmap=cmap_gt, alpha=alpha)
    ax.imshow(img_pred, cmap=cmap_pred, alpha=alpha)
    cmap_pred = ListedColormap(['none', colors[label2]])
    
    if add_eval:
        dice_score = dice(img_pred, img_gt)
        overlap = overlap_perc(img_pred, img_gt)
        ax.set_title('Dice: '+str(round(dice_score,2))+'\n Overlap: '+str(int(overlap))+' %'+add_title)
        
    if add_legend:
        patches =[mpatches.Patch(color=colors[label],label=label) for label in labels]
        ax.legend(handles=patches, loc=2, borderaxespad=0., bbox_to_anchor=(1.02,1))
                   
    # save and/or display plot 
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()

def get_metric_range(g=None, metric='corr'):
    ''' Get range of the metric
    g: seaborn plot object
    metric: metric name {corr, ssim, other}
    '''
    if metric == 'corr':
        rng = [-1, 1]
    elif metric == 'ssim':
        rng = [-1, 1]
    else:
        assert g is not None, 'Unable to extract range as no plot object provided!'
        xmin, xmax = g.get_xlim()
        ymin, ymax = g.get_ylim()
        rng = [min(xmin, ymin), max(xmax,ymax)]
    return rng

def plot_metric_scatter(x, y, data, hue=None, size=None, metric='corr', xlab=None, ylab=None, 
                        figsize=(6,4), save_fname=None, plot_show=False):
    ''' Plot a scatterplot with diagonal line and color/hue encoding
    x: column name to plot on xaxis (eval measure for modelA)
    y: column name to plot on yaxis (eval measure for modelB)
    data: dataframe with columns {x,y,hue,size}
    '''
    plt.figure(figsize=figsize)
    g = sns.scatterplot(x=x, y=y, data=data, hue=hue, size=size)
    # add diagonal
    rng = get_metric_range(g, metric=metric)
    plt.plot(rng, rng, color='lightgrey', linestyle = '--')
    # set title and axes labels
    pr = round(data.corr(method='pearson').loc[x, y],2)
    plt.title('Pearson: '+str(pr))
    if xlab is not None:
        g.set_xlabel(xlab)
    if ylab is not None:
        g.set_ylabel(ylab)        
    plt.legend(bbox_to_anchor=(1,1))
    # save and/or display plot 
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()


def plot_gt_top(pred_img_path, protein_set, top_epochs, figsize=(20,14), save_fname=None, plot_show=False):
    '''Plot ground truth and predictions for top epochs
    pred_img_path: absolute path to the folder containing predicted images
    protein_set: list of proteins corresponding to channels in predicted images
    top_epochs: list of names of top epochs (e.g., ['epoch29'])
    '''

    pred_img_path = Path(pred_img_path) #model_path.joinpath(model_name, 'valid_images')
    rois = os.listdir(pred_img_path.joinpath(os.listdir(pred_img_path)[0]))
    for roi in rois:
        print(roi)
        nrows = len(protein_set)
        ncols = len(top_epochs)+2
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        gt_he = np.load(Path(BINARY_HE_ROI_STORAGE).joinpath(roi.split('-')[-1]))
        gt = np.load(Path(BINARY_IMC_ROI_STORAGE).joinpath(roi.split('-')[-1]))

        j = 2
        for epoch in top_epochs:
            i = 0
            pred = np.load(pred_img_path.joinpath(epoch, roi))
            for idx, protein in enumerate(protein_set):
                protein_idx = get_protein_idx(len(protein_set), protein)
                gt_prot = gt[:,:,protein_idx]
                pred_prot = pred[:,:,idx]
                assert gt_prot.shape == pred_prot.shape, 'Shapes of GT and predicted images do not match!'
                if len(protein_set)==1:
                    axes[0].imshow(gt_he)
                    axes[0].set_title('HE')
                    axes[0].set_yticks([])
                    axes[0].set_xticks([])
                    axes[1].imshow(gt_prot)
                    axes[1].set_title('GT: '+protein)
                    axes[1].set_yticks([])
                    axes[1].set_xticks([])
                    axes[j].imshow(pred_prot)
                    axes[j].set_title(epoch+': '+protein)
                    axes[j].set_yticks([])
                    axes[j].set_xticks([])
                else:
                    axes[i,0].imshow(gt_he)
                    axes[i,0].set_title('HE')
                    axes[i,0].set_yticks([])
                    axes[i,0].set_xticks([])
                    axes[i,1].imshow(gt_prot)
                    axes[i,1].set_title('GT: '+protein)
                    axes[i,1].set_yticks([])
                    axes[i,1].set_xticks([])
                    axes[i,j].imshow(pred_prot)
                    axes[i,j].set_title(epoch+': '+protein)
                    axes[i,j].set_yticks([])
                    axes[i,j].set_xticks([])
                    i = i+1
            j = j+1
        if save_fname is not None:
            plt.savefig(str(save_fname).replace('.png', '-'+roi.replace('.npy','.png')), dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show()
        
def plot_ct_top(pred_img_path, fname, types, sel_types, top_epochs, figsize=(14,14), save_fname=None, plot_show=False):
    '''Plot ground truth and predictions for top epochs
    pred_img_path: absolute path to the folder containing predicted images
    fname: file name (rf predictions for predicted images)
    type: list of cell-types corresponding to channels in predicted images
    top_epochs: list of names of top epochs (e.g., ['epoch29'])
    '''
    roi = fname.split('-')[-1]
    pred_img_path = Path(pred_img_path)
    nrows = len(sel_types)
    ncols = len(top_epochs)+2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    gt_he = np.load(Path(BINARY_HE_ROI_STORAGE).joinpath(roi))
    gt = np.load(Path(pred_img_path).joinpath(top_epochs[0],fname.replace('rf_pred','rf_gt_pred')))

    j = 2
    for epoch in top_epochs:
        i = 0
        pred = np.load(pred_img_path.joinpath(epoch, fname))
        for ct in sel_types:
            idx = [i for i,x in enumerate(types) if x == ct]
            gt_prot = gt[:,:,idx]
            pred_prot = pred[:,:,idx]
            assert gt_prot.shape == pred_prot.shape, 'Shapes of GT and predicted images do not match!'
            if len(types)==1:
                axes[0].imshow(gt_he)
                axes[0].set_title('HE')
                axes[0].set_yticks([])
                axes[0].set_xticks([])
                axes[1].imshow(gt_prot)
                axes[1].set_title('GT: '+ct)
                axes[1].set_yticks([])
                axes[1].set_xticks([])
                axes[j].imshow(pred_prot)
                axes[j].set_title(epoch+': '+ct)
                axes[j].set_yticks([])
                axes[j].set_xticks([])
            else:
                axes[i,0].imshow(gt_he)
                axes[i,0].set_title('HE')
                axes[i,0].set_yticks([])
                axes[i,0].set_xticks([])
                axes[i,1].imshow(gt_prot)
                axes[i,1].set_title('GT: '+ct)
                axes[i,1].set_yticks([])
                axes[i,1].set_xticks([])
                axes[i,j].imshow(pred_prot)
                axes[i,j].set_title(epoch+': '+ct)
                axes[i,j].set_yticks([])
                axes[i,j].set_xticks([])
                i = i+1
        j = j+1
    if save_fname is not None:
        plt.savefig(str(save_fname).replace('.png', '-'+fname.replace('.npy','.png')), dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()