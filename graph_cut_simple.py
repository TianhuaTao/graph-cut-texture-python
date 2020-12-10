from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
import maxflow
from random import random, randint
import os

if not os.path.exists('out/graph_cut_simple'):
    os.makedirs('out/graph_cut_simple')


class GraphCutTexture():
    def __init__(self, input_img, output_height, output_width):
        # src img
        self.input_img = input_img
        # src img mask, all ones, shape=(input_height, input_weight)
        self.input_img_mask = np.ones((input_img.shape[0], input_img.shape[1]), dtype=int)
        self.input_height = input_img.shape[0]
        self.input_width = input_img.shape[1]

        self.output_height = output_height
        self.output_width = output_width

        self.output_img = np.zeros((output_height, output_width, 3), dtype=int)
        self.output_img_filled_mask = np.zeros((output_height, output_width), dtype=int)
        self.patch_number = 0

        plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')

    def insertPatch(self, y: int, x: int):
        print('patch ', self.patch_number)
        new_y = y if y >= 0 else 0
        new_x = x if x >= 0 else 0
        new_height = self.input_height
        new_width = self.input_width
        if y < 0:
            new_height += y
        if x < 0:
            new_width += x
        if new_y + new_height >= self.output_height:
            new_height -= new_y + new_height - self.output_height + 1
        if new_x + new_width >= self.output_width:
            new_width -= new_x + new_width - self.output_width + 1

        # construct new patch mask in a full matrix
        input_img_mask_expanded = np.zeros_like(self.output_img_filled_mask)
        self.copy_to_offset(input_img_mask_expanded, self.input_img_mask, (y, x))
        self.set_img(input_img_mask_expanded, 2, 'input_img_mask_expanded')

        # get the overlap area between new patch and filled area
        overlap_mask = input_img_mask_expanded * self.output_img_filled_mask

        # if the new patch has no overlapping area, direct copy
        if not overlap_mask.any():
            self.copy_to_offset(self.output_img, self.input_img, (y, x))
            self.output_img_filled_mask |= input_img_mask_expanded
            self.patch_number += 1
        else:
            # do a graph cut insertion

            # copy the new patch to a big empty buffer
            new_patch_buffer = np.zeros_like(self.output_img)
            self.copy_to_offset(new_patch_buffer, self.input_img, (y, x))

            new_patch_pixel_count_estimated = input_img_mask_expanded.sum()

            output_img_cropped_height = (input_img_mask_expanded.sum(axis=1) > 0).astype(int).sum()
            assert output_img_cropped_height == new_height
            output_img_cropped_width = (input_img_mask_expanded.sum(axis=0) > 0).astype(int).sum()
            assert output_img_cropped_width == new_width

            old_patch_cropped = self.output_img[input_img_mask_expanded > 0].reshape(
                output_img_cropped_height,
                output_img_cropped_width,
                -1
            )
            new_patch_cropped = new_patch_buffer[input_img_mask_expanded > 0].reshape(
                output_img_cropped_height,
                output_img_cropped_width,
                -1
            )

            # make graph
            g = maxflow.Graph[int](new_patch_pixel_count_estimated, new_patch_pixel_count_estimated)

            # make nodes
            nodeids = g.add_grid_nodes((new_patch_cropped.shape[0], new_patch_cropped.shape[1]))

            # make edges
            cost_matrix_right = self.construct_cost_matrix_right(
                new_patch_cropped,
                old_patch_cropped
            )
            cost_matrix_down = self.construct_cost_matrix_down(
                new_patch_cropped,
                old_patch_cropped
            )

            # add right and left
            structure = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]])

            g.add_grid_edges(nodeids, weights=cost_matrix_right, structure=structure,
                             symmetric=True)

            # add down and up
            structure = np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 1, 0]])
            g.add_grid_edges(nodeids, weights=cost_matrix_down, structure=structure,
                             symmetric=True)

            # add terminal edges

            old_mask_cropped = self.get_cropped(self.output_img_filled_mask,
                                                offset=(y, x),
                                                shape=(self.input_height, self.input_width)
                                                )
            assert old_mask_cropped.shape[0] == new_height
            assert old_mask_cropped.shape[1] == new_width

            self.set_img(old_mask_cropped, 5, 'old_mask_cropped')

            nodeids_connected_to_old, nodeids_connected_to_new = self.get_nodeids_connected_to_old_and_new(nodeids,
                                                                                                           old_mask_cropped)

            inf_weight = np.ones_like(nodeids_connected_to_old) * 90000  # very big number
            g.add_grid_tedges(nodeids_connected_to_old, inf_weight, 0)

            inf_weight = np.ones_like(nodeids_connected_to_new) * 90000  # very big number
            g.add_grid_tedges(nodeids_connected_to_new, 0, inf_weight)

            # Find the maximum flow.
            flow = g.maxflow()

            # Get the segments of the nodes in the grid.
            sgm = g.get_grid_segments(nodeids)
            # print(sgm.sum())

            self.set_img(sgm, 6, 'sgm')

            overlap_buffer = np.array(old_patch_cropped)
            overlap_buffer[sgm] = new_patch_cropped[sgm]

            self.copy_to_offset(self.output_img, new_patch_cropped, (new_y, new_x))
            self.copy_to_offset(self.output_img, overlap_buffer, (new_y, new_x))

            self.output_img_filled_mask |= input_img_mask_expanded

            # self.show_output_img()
            self.patch_number += 1

    def get_nodeids_connected_to_old_and_new(self, nodeids, old_mask):
        connected_to_new_mask = (old_mask == 0)

        connected_to_old_mask = np.ones_like(old_mask)
        connected_to_old_mask[1:-1, 1:-1] = 0
        connected_to_old_mask &= old_mask
        self.set_img(connected_to_old_mask, 3, 'connected_to_old')
        self.set_img(connected_to_new_mask, 4, 'connected_to_new')
        assert (connected_to_old_mask * connected_to_new_mask).sum() == 0

        if not connected_to_new_mask.any():
            yy = connected_to_new_mask.shape[0] // 2
            xx = connected_to_new_mask.shape[1] // 2
            connected_to_new_mask[yy, xx] = True
            connected_to_old_mask[yy, xx] = False

        connected_to_new = nodeids[connected_to_new_mask]
        connected_to_old = nodeids[connected_to_old_mask > 0]

        assert (connected_to_new_mask.any())
        assert (connected_to_old_mask.any())

        return connected_to_old, connected_to_new

    def construct_cost_matrix_right(self, overlap_new: np.array, overlap_old: np.array):
        difference_between_patch = np.abs(overlap_new - overlap_old)
        shift_left_dif = np.roll(difference_between_patch, (0, -1))
        match_cost_right = (difference_between_patch + shift_left_dif).sum(axis=2)
        return match_cost_right + 1

    def construct_cost_matrix_down(self, overlap_new: np.array, overlap_old: np.array):
        difference_between_patch = np.abs(overlap_new - overlap_old)
        shift_up_dif = np.roll(difference_between_patch, (-1, 0))
        match_cost_down = (difference_between_patch + shift_up_dif).sum(axis=2)
        return match_cost_down + 1

    def get_cropped(self, src: np.ndarray, offset: Tuple[int, int], shape: Tuple[int, int]):
        offset_y, offset_x = offset
        crop_height, crop_width = shape
        if offset_y < 0:  # out of top bound
            crop_height += offset_y
            offset_y = 0
        if offset_x < 0:  # out of left bound
            crop_width += offset_x
            offset_x = 0
        if offset_y + crop_height >= src.shape[0]:  # out of bottom bound
            remain = offset_y + crop_height - src.shape[0] + 1
            crop_height -= remain
        if offset_x + crop_width >= src.shape[1]:  # out of right bound
            remain = offset_x + crop_width - src.shape[1] + 1
            crop_width -= remain

        return src[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width]

    def copy_to_offset(self, dst: np.ndarray, src: np.ndarray, dst_offset: Tuple[int, int]):
        src_height = src.shape[0]
        src_width = src.shape[1]
        dst_offset_y, dst_offset_x = dst_offset
        src_offset_y = 0
        src_offset_x = 0
        if dst_offset_y < 0:
            src_offset_y += -dst_offset_y
            src_height += dst_offset_y
            dst_offset_y = 0
        if dst_offset_x < 0:
            src_offset_x += -dst_offset_x
            src_width += dst_offset_x
            dst_offset_x = 0
        if dst_offset_y + src_height >= dst.shape[0]:
            remain = dst_offset_y + src_height - dst.shape[0] + 1
            src_height -= remain
        if dst_offset_x + src_width >= dst.shape[1]:
            remain = dst_offset_x + src_width - dst.shape[1] + 1
            src_width -= remain

        dst[dst_offset_y:dst_offset_y + src_height, dst_offset_x:dst_offset_x + src_width] = \
            src[src_offset_y:src_offset_y + src_height, src_offset_x:src_offset_x + src_width]

    def random_fill(self):
        print('Initial synthesis: Random')
        overlap_width = self.input_width // 3
        overlap_height = self.input_height // 3

        offset_y = 0
        x = 0
        y = offset_y - (overlap_height + randint(0, overlap_height - 1))

        while True:
            print('New Row')
            x = -(overlap_width + randint(0, overlap_width - 1))
            while True:

                if y < self.output_height:
                    self.insertPatch(y, x)

                x = x + (overlap_width + randint(0, overlap_width - 1))
                y = offset_y - (overlap_height + randint(0, overlap_height - 1))

                # gc_texture.save_fig(self.patch_number)

                if x >= self.output_width:
                    break

            offset_y += overlap_height
            y = offset_y - (overlap_height + randint(0, overlap_height - 1))

            if y >= self.output_height:
                break

    def show_output_img(self):
        plt.subplot(1, 1, 1)
        plt.imshow(self.output_img)
        plt.show()

    def set_img(self, img, index, title=None):
        # ax = plt.subplot(3, 2, index)
        # ax.set_title(title)
        # plt.imshow(img)
        pass

    def save_fig(self, index):
        name = 'out/graph_cut_simple/out_{}.png'.format(index)
        plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.output_img)
        plt.savefig(name)


if __name__ == "__main__":
    img_in = imread('data/strawberries2.gif')
    # img_in = imread('data/green.gif')
    # img_in = imread('data/akeyboard_small.gif')
    if img_in.shape[2] == 4:
        # remove alpha channel
        img_in = np.array(img_in[:, :, 0:3])
    # plt.imshow(img_in)
    # plt.show()

    print('original image size: ', img_in.shape)

    gc_texture = GraphCutTexture(img_in, img_in.shape[0] * 2, img_in.shape[1] * 2)

    gc_texture.random_fill()
    gc_texture.show_output_img()
