from imageio import imread
import numpy as np
import scipy
from matplotlib import pyplot as plt
from typing import Tuple
import maxflow

plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')


def copy_to_offset(dst: np.ndarray, src: np.ndarray, offset:tuple):
    src_height = src.shape[0]
    src_width = src.shape[1]
    offset_y, offset_x = offset
    dst[offset_y:offset_y+src_height,offset_x:offset_x+src_width] = src


def test1(img_in:np.array):
    input_height = img_in.shape[0]
    input_width = img_in.shape[1]
    output_height = input_height*3
    output_width = input_width*3

    img_out = np.zeros(( output_height,output_width, 3),dtype=int)

    
    
    new_patch = np.array(img_in)
    # plt.imshow(new_patch)
    # plt.show()
    copy_to_offset(dst=img_out, src=new_patch, offset=(0,0))

    copy_to_offset(dst=img_out, src=new_patch, offset=(0,input_width-10))

    plt.imshow(img_out)
    plt.show()


def get_overlap_mask(dst:np.array, dst_mask:np.array ,new_patch:np.array, new_patch_offset: tuple):
    pass

def match_cost(pix_offset1: Tuple[int, int], pix_offset2:Tuple[int, int], old_patch:np.array, new_patch:np.array):
    p1o = old_patch[pix_offset1]
    p2o = old_patch[pix_offset2]
    p1n = new_patch[pix_offset1]
    p2n = new_patch[pix_offset2]
    c1 = p1o-p1n
    c2 = p2o-p2n
    c = np.abs(c1)+np.abs(c2)
    c = c.sum()
    return c

def construct_cost_matrix_right(overlap_new:np.array, overlap_old:np.array):
    difference_between_patch = np.abs(overlap_new-overlap_old)
    shift_left_dif = np.roll(difference_between_patch, (0, -1))
    match_cost_right = (difference_between_patch+ shift_left_dif).sum(axis =2)
    return  match_cost_right

def construct_cost_matrix_down(overlap_new:np.array, overlap_old:np.array):
    difference_between_patch = np.abs(overlap_new-overlap_old)
    shift_up_dif = np.roll(difference_between_patch, (-1, 0))
    match_cost_down = (difference_between_patch+ shift_up_dif).sum(axis =2)
    return match_cost_down

## test construct_cost_matrix_right
# A= np.array([
#     [1,5,7],
#     [2,3,3],
#     [4,0,10],
# ])
# B= np.array([
#     [3,4,7],
#     [2,4,7],
#     [1,5,9],
# ])
# print(construct_cost_matrix_right(A, B))
# print(construct_cost_matrix_right(B, A))

def add_patch(dst:np.array, dst_mask:np.array ,new_patch:np.array, new_patch_offset: tuple):
    # copy_to_offset(dst=dst, src=new_patch, offset=new_patch_offset)
    # copy_to_offset(dst=dst_mask, src=np.ones((new_patch.shape[0], new_patch.shape[1]), dtype=int), offset=new_patch_offset)    # add mask

    # construct new patch buffer
    new_patch_buffer = np.zeros_like(dst)
    copy_to_offset(new_patch_buffer, new_patch, new_patch_offset)

    # construct new patch mask
    new_patch_mask_relative = np.ones((new_patch.shape[0], new_patch.shape[1]), dtype=int)
    new_patch_mask_absolute = np.zeros((new_patch_buffer.shape[0], new_patch_buffer.shape[1]), dtype=int)
    copy_to_offset(dst=new_patch_mask_absolute, src=new_patch_mask_relative, offset=new_patch_offset)    # add mask

    # construct overlap mask
    overlap_mask = (new_patch_mask_absolute * dst_mask)
    overlap_height = (overlap_mask.sum(axis=1)>0).astype(int).sum()
    overlap_width = (overlap_mask.sum(axis=0)>0).astype(int).sum()
    overlap_mask = overlap_mask>0


    num_overlap_pixels = new_patch_mask_absolute.sum()
    overlap_new = new_patch_buffer[overlap_mask].reshape(overlap_height, overlap_width, -1)
    overlap_old = dst[overlap_mask].reshape(overlap_height, overlap_width, -1)
    cost_matrix_right = construct_cost_matrix_right(overlap_new, overlap_old)
    cost_matrix_down = construct_cost_matrix_down(overlap_new, overlap_old)

    g = maxflow.Graph[int](num_overlap_pixels,num_overlap_pixels)
    nodeids = g.add_grid_nodes((overlap_height, overlap_width))
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

    left_most = nodeids[:, :1]
    right_most = nodeids[:,-1:]
    # up_most = nodeids[:1, :]
    # down_most = nodeids[-1:, :]
    inf_weight = np.ones_like(left_most)*90000 # very big number
    g.add_grid_tedges(left_most, inf_weight, 0)
    g.add_grid_tedges(right_most, 0, inf_weight)

    # Find the maximum flow.
    flow = g.maxflow()
    print('flow', flow)
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)


    tmp = sgm[:,40:]
    print(tmp.sum())
    print(tmp.shape)


    overlap_buffer = np.array(overlap_old)
    overlap_buffer[sgm] = overlap_new[sgm]

    copy_to_offset(dst=dst, src=new_patch, offset=new_patch_offset)
    copy_to_offset(dst=dst, src=overlap_buffer, offset=new_patch_offset)

    merge_mask = np.zeros_like(dst_mask)
    copy_to_offset(dst=merge_mask, src=new_patch_mask_relative, offset=new_patch_offset)
    copy_to_offset(dst=merge_mask, src=sgm, offset=new_patch_offset)

    plt.subplot(2, 1, 2)
    plt.imshow(merge_mask)

def test2(img_in:np.array):
    overlap_width =  100
    input_height = img_in.shape[0]
    input_width = img_in.shape[1]
    result_height = input_height*3
    result_width = input_width*3

    result_img = np.zeros(( result_height,result_width, 3),dtype=int)
    result_img_mask = np.zeros(( result_height,result_width),dtype=int)

    new_patch = np.array(img_in)
    new_patch_mask = np.ones((img_in.shape[0], img_in.shape[1]), dtype=int)

    # first patch, place at left top
    copy_to_offset(dst=result_img, src=new_patch, offset=(0,0))
    copy_to_offset(dst=result_img_mask, src=new_patch_mask, offset=(0,0))    # add mask


    new_patch_offset = (0, new_patch.shape[1]-overlap_width)
    add_patch(result_img, result_img_mask, new_patch, new_patch_offset)

    plt.subplot(2,1,1)
    plt.imshow(result_img)

    plt.show()

if __name__ == "__main__":
    img_in = imread('data/strawberries2.gif')
    if img_in.shape[2] == 4:
        # remove alpha channel
        img_in = img_in[:,:,0:3]
    # plt.imshow(img_in)
    # plt.show()

    print('original image size: ', img_in.shape)

    test2(img_in)
