import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('positions_measured.pkl', 'rb') as f:
    positions_measured = pickle.load(f)
with open('cluster_label_history.pkl', 'rb') as f:
    cluster_label_history = pickle.load(f)

data = xr.open_dataset('data.nc')
data = data['ARPES'][:len(positions_measured)].values
counts = data.sum(axis=1).sum(axis=1)
from PIL import Image

def fig_to_pil(fig):
    fig.canvas.draw()
    return Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

def order_cluster_labels_by_counts(positions_measured, labels, counts):
    positions_measured = np.asarray(positions_measured)
    labels = np.asarray(labels)
    counts = np.asarray(counts)[:len(labels)]

    num_pts = positions_measured.shape[0]
    counts = counts[:num_pts]
    median_counts = []
    labels_looped = []
    # print(labels.shape, counts.shape)
    for i, label in enumerate(set(labels)):
        median_counts.append(np.sum(np.where(labels == label, counts, 0)))
        labels_looped.append(label)
    
    ordered_labels = np.zeros(labels.shape)*np.nan
    indices = np.argsort(median_counts)
    # median_counts_ordered = np.array(median_counts)[indices]
    labels_ordered = np.array(labels_looped)[indices]
    # print(median_counts_ordered)
    # print(labels_looped)

    lookup_new_index = {before: after for after, before in zip(labels_looped, labels_ordered)}
    ordered_labels = [lookup_new_index.get(item,item) for item in labels]
    return ordered_labels

from scipy.interpolate import griddata
minx, maxx = np.min(positions_measured, axis=0)[0], np.max(positions_measured, axis=0)[0]
miny, maxy = np.min(positions_measured, axis=0)[1], np.max(positions_measured, axis=0)[1]
xgrid = np.linspace(minx, maxx, 200)
ygrid = np.linspace(miny, maxy, 200)
xgrid, ygrid = np.meshgrid(xgrid, ygrid)

print(np.min(xgrid.flatten()), np.max(xgrid.flatten()), np.min(ygrid.flatten()), np.max(ygrid.flatten()))


imgs = []
num = len(positions_measured)
# for i in range(4,num+1):
for i in range(1,num+1):
    if i < 4:
        counts_grid = np.zeros(xgrid.shape)
    else:
        counts_grid = griddata(positions_measured[:i], counts[:i], (xgrid, ygrid), method='cubic')
        counts_grid -= np.nanmin(counts_grid)
        counts_grid /= np.nanmax(counts_grid)
        counts_grid = np.nan_to_num(counts_grid)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(f'Measurement: {i}    Number of Clusters: {1+max(set(cluster_label_history[i-1]))}')
#     fig, (ax1,ax3) = plt.subplots(1,2, figsize=(10,5))
    labels_ordered = order_cluster_labels_by_counts(positions_measured, cluster_label_history[i-1], counts)
    label_nearest_matrix = np.round(griddata(np.array(positions_measured[:i]), labels_ordered, (xgrid, ygrid), method='nearest'))
    ax1.imshow(label_nearest_matrix, extent=[minx,maxx,miny,maxy], origin='lower', cmap='tab20')
    ax1.scatter(*np.array(positions_measured[:i]).T, marker='x', c='k', s=1)
    ax2.imshow(counts_grid, extent=[minx,maxx,miny,maxy], origin='lower', cmap='gray_r')
#     ax2.scatter(*np.array(positions_measured[:i]).T, marker='x', c='k', s=1)
    ax3.imshow(data[i-1], origin='lower', cmap='gray_r')
#     imgs.append(fig_to_pil(fig))
    plt.savefig(f'movie/{i:04d}.png')
    plt.close()
    print(f'{i/num:.2%}%', end='\r')

from compile_pngs import ffmpeg_compile_pngs_to_mp4
ffmpeg_compile_pngs_to_mp4()

