from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
import maxflow
from random import random, randint
import os

# name = 'boat'
name = 'hut'
source_color = [0, 128, 255]
sink_color = [255, 255, 0]

plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')


def construct_cost_matrix_right(new: np.ndarray, old: np.ndarray):
    difference_between_patch = np.abs(new - old)
    shift_left_dif = np.roll(difference_between_patch, (0, -1))
    match_cost_right = (difference_between_patch + shift_left_dif).sum(axis=2)
    return match_cost_right + 1


def construct_cost_matrix_down(new: np.ndarray, old: np.ndarray):
    difference_between_patch = np.abs(new - old)
    shift_up_dif = np.roll(difference_between_patch, (-1, 0))
    match_cost_down = (difference_between_patch + shift_up_dif).sum(axis=2)
    return match_cost_down + 1


def graph_cut_blend(src_img, tgt_img, mask):
    if src_img.shape != tgt_img.shape:
        print('Error')
    img_height, img_width, _ = src_img.shape

    g = maxflow.Graph[int](img_height, img_width)
    nodeids = g.add_grid_nodes((img_height, img_width))

    cost_matrix_right = construct_cost_matrix_right(src_img, tgt_img)
    cost_matrix_down = construct_cost_matrix_down(src_img, tgt_img)
    # add right
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

    g.add_grid_edges(nodeids, weights=cost_matrix_right, structure=structure,
                     symmetric=True)

    # add down
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    g.add_grid_edges(nodeids, weights=cost_matrix_down, structure=structure,
                     symmetric=True)

    sink_node = []
    source_node = []
    inf_weight = 90000  # very big number

    for j in range(img_height):
        for i in range(img_width):
            # if (mask[j,i] != 0).any():
            #     print('yes')
            #     print(mask[j,i])
            if np.equal(mask[j, i], source_color).all():
                nodeid = nodeids[j, i]
                source_node.append(nodeid)
                g.add_tedge(nodeid, inf_weight, 0)

            elif np.equal(mask[j, i], sink_color).all():
                nodeid = nodeids[j, i]
                sink_node.append(nodeid)
                g.add_tedge(nodeid, 0, inf_weight)

    # Find the maximum flow.
    flow = g.maxflow()
    print('flow', flow)
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    print(sgm)

    tgt_img[sgm] = src_img[sgm]
    plt.subplot(2, 1, 1)
    plt.imshow(tgt_img)
    plt.subplot(2, 1, 2)
    plt.imshow(sgm)

    plt.show()

    return tgt_img


if __name__ == "__main__":
    src_img_in = imread('data/{}_src.jpg'.format(name))
    tgt_img_in = imread('data/{}_target.jpg'.format(name))
    mask_img = imread('data/{}_mask.png'.format(name))

    if src_img_in.shape[2] == 4:
        # remove alpha channel
        src_img_in = np.array(src_img_in[:, :, 0:3])
    if tgt_img_in.shape[2] == 4:
        # remove alpha channel
        tgt_img_in = np.array(tgt_img_in[:, :, 0:3])

    out_img = graph_cut_blend(src_img_in, tgt_img_in, mask_img)
