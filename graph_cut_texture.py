from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

import maxflow
import random
from random import randint
import scipy.ndimage as nd
import os
from scipy import signal

# random.seed(0)

plt.figure(num=None, figsize=(40, 32), dpi=80, facecolor='w', edgecolor='k')

minCap = 1e-7
infiniteCap = 1e12
source_type = 0
sink_type = 1
out_dir = ''  # assigned in main
data_filename = ''  # assigned in main


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def square_distance(c1, c2):
    diff = (c1 / 255.0 - c2 / 255.0)
    return (diff * diff).sum()


def abs_distance(c1, c2):
    diff = c1 / 255.0 - c2 / 255.0
    dis = np.sqrt((diff * diff).sum())
    return dis


def compute_gradient_image(image):
    x_kernel = np.array([[0, 0, 0],
                         [1, 0, -1],
                         [0, 0, 0]])
    y_kernel = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]])
    image_gray = rgb2gray(image)
    grad_x = nd.convolve(image_gray, x_kernel, mode='constant')
    grad_y = nd.convolve(image_gray, y_kernel, mode='constant')
    grad_x = np.abs(grad_x)
    grad_y = np.abs(grad_y)

    return grad_y, grad_x


class GlobalNode:
    def __init__(self):
        self.empty = True
        self.rightCost = 0.0
        self.bottomCost = 0.0
        self.seamRight = False
        self.seamBottom = False
        self.maxFlow = 0.0
        self.newSeam = False
        self.color = np.zeros(3, dtype=int)
        self.colorOtherPatch = np.zeros(3, dtype=int)
        self.gradXOtherPatch = 0
        self.gradYOtherPatch = 0


class SeamNode:
    def __init__(self, start, end, c1, c2, c3, orientation):
        self.start = start
        self.end = end
        self.capacity1 = c1
        self.capacity2 = c2
        self.capacity3 = c3
        self.orientation = orientation
        self.seam = 0


class GraphCutTexture:
    def __init__(self, input_img, output_height, output_width):
        # src img
        self.input_img = input_img
        # src img mask, all ones, shape=(input_height, input_weight)
        self.input_img_mask = np.ones((input_img.shape[0], input_img.shape[1]), dtype=int)
        self.input_height = input_img.shape[0]
        self.input_width = input_img.shape[1]

        self.croppedInputImage = None
        self.cropped_input_w = 0
        self.cropped_input_h = 0
        self.croppedInputImageGX = None
        self.croppedInputImageGY = None

        self.output_height = output_height
        self.output_width = output_width

        self.output_img = np.zeros((output_height, output_width, 3), dtype=int)
        self.output_img_filled_mask = np.zeros((output_height, output_width), dtype=int)
        self.patch_number = 0

        self.global_nodes = [GlobalNode() for _ in range(output_width * output_height)]
        self.seamNode: [SeamNode] = []

        self.inputImageGY, self.inputImageGX = compute_gradient_image(self.input_img)
        self.borderSize = 16

        self.maxErrNodeNbGlobal = -1
        self.num_sink = 0
        self.sig = (self.input_img.sum(axis=2) / 3.0 / 255.0).std()
        self.sig2 = self.sig * self.sig

        self.cropped_left_top_input_space = None
        self.cropped_right_bottom_input_space = None
        self.cropped_left_top_output_space = None
        self.cropped_right_bottom_output_space = None

        self.linked_to_sink = set()
        self.linked_to_source = set()

        self.summed_area_table_i_squared = self.calculate_summed_table(self.input_img)

        self.used_offset = set()

    def calculate_summed_table(self, img):
        input_avg = img.sum(axis=2) / 3 / 255.0
        i_2 = input_avg * input_avg
        return i_2.cumsum(axis=0).cumsum(axis=1)

    def pick_error_region(self, radius):
        if self.maxErrNodeNbGlobal == -1:
            print('Error: no error region')
            return
        max_err_y, max_err_x = self.get_position_global(self.maxErrNodeNbGlobal)
        v1_y = max_err_y - radius
        v1_x = max_err_x - radius
        v2_y = max_err_y + radius
        v2_x = max_err_x + radius
        v1_y = max(v1_y, 0)
        v1_x = max(v1_x, 0)
        v2_y = min(v2_y, self.output_height)
        v2_x = min(v2_x, self.output_width)

        return v1_y, v1_x, v2_y, v2_x

    def count_non_overlap(self, x, y):
        no_overlap = 0
        for j in range(self.cropped_input_h):
            for i in range(self.cropped_input_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                if self.global_nodes[node_nb_global].empty:
                    no_overlap += 1
        return no_overlap

    def direct_copy_input_to_output(self, y, x):
        for j in range(self.cropped_input_h):
            for i in range(self.cropped_input_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                the_node = self.global_nodes[node_nb_global]
                if the_node.empty:
                    the_node.color = self.croppedInputImage[j, i]
                    the_node.empty = False

    def compute_cropped_output_image(self, y, x):
        image = np.zeros_like(self.croppedInputImage)
        for j in range(self.cropped_input_h):
            for i in range(self.cropped_input_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                if not self.global_nodes[node_nb_global].empty:
                    image[j, i] = self.global_nodes[node_nb_global].color
                else:
                    image[j, i] = [0, 0, 0]
        return image

    def configure_node_capacity(self, y, x, g, image_gy, image_gx):
        overlap: int = 0
        for j in range(self.cropped_input_h):
            for i in range(self.cropped_input_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                node_nb_local = self.get_node_number_local(i, j)

                if i < self.cropped_input_w - 1:
                    node_nb_global_right = self.get_node_number_global(x, y, i + 1, j)
                    node_nb_local_right = self.get_node_number_local(i + 1, j)

                    if not self.global_nodes[node_nb_global].empty:
                        if not self.global_nodes[node_nb_global_right].empty:  # right
                            d1 = abs_distance(self.global_nodes[node_nb_global].color, self.croppedInputImage[j, i])
                            d2 = abs_distance(self.global_nodes[node_nb_global_right].color,
                                              self.croppedInputImage[j, i + 1])
                            if self.global_nodes[node_nb_global].seamRight:
                                # Old seam: a seam node will created
                                capacity1 = self.global_nodes[node_nb_global].rightCost
                                d3 = abs_distance(self.global_nodes[node_nb_global_right].colorOtherPatch,
                                                  self.croppedInputImage[j, i + 1])
                                d4 = abs_distance(self.croppedInputImage[j, i],
                                                  self.global_nodes[node_nb_global].colorOtherPatch)
                                grad = ((self.croppedInputImageGX[j, i] / 255.0)
                                        + (self.croppedInputImageGX[j, i + 1] / 255.0)
                                        + (image_gx[j, i] / 255.0)
                                        + (self.global_nodes[node_nb_global_right].gradXOtherPatch / 255.0))
                                grad += 1.0

                                capacity2 = (d1 + d3) / grad
                                grad = (self.croppedInputImageGX[j, i] / 255.0) \
                                       + (self.croppedInputImageGX[j, i + 1] / 255.0) \
                                       + (image_gx[j, i + 1] / 255.0) \
                                       + (self.global_nodes[node_nb_global].gradXOtherPatch / 255.0)
                                grad += 1.0

                                capacity3 = (d4 + d2) / grad
                                capacity2 += minCap
                                capacity3 += minCap
                                self.seamNode.append(
                                    SeamNode(node_nb_local, node_nb_local_right, capacity1, capacity2, capacity3, 0))
                            else:
                                # No old seam
                                grad = (self.croppedInputImageGX[j, i] / 255.0) \
                                       + (self.croppedInputImageGX[j, i + 1] / 255.0) \
                                       + (image_gx[j, i] / 255.0) \
                                       + (image_gx[j, i + 1] / 255.0)
                                grad += 1.0
                                capRight = (d1 + d2) / grad
                                capRight += minCap
                                g.add_edge(node_nb_local, node_nb_local_right, capRight, capRight)
                                self.global_nodes[node_nb_global].rightCost = capRight
                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(node_nb_local, node_nb_local_right, 0.0, 0.0)
                            self.global_nodes[node_nb_global].rightCost = 0.0
                    else:
                        g.add_edge(node_nb_local, node_nb_local_right, 0.0, 0.0)
                        self.global_nodes[node_nb_global].rightCost = 0.0

                if j < self.cropped_input_h - 1:
                    nodeNbGlobalBottom = self.get_node_number_global(x, y, i, j + 1)
                    nodeNbLocalBottom = self.get_node_number_local(i, j + 1)

                    if not self.global_nodes[node_nb_global].empty:
                        if not self.global_nodes[nodeNbGlobalBottom].empty:  # bottom
                            # Overlap
                            d1 = abs_distance(self.global_nodes[node_nb_global].color, self.croppedInputImage[j, i])
                            d2 = abs_distance(self.global_nodes[nodeNbGlobalBottom].color,
                                              self.croppedInputImage[j + 1, i])
                            if self.global_nodes[node_nb_global].seamBottom:
                                # Old seam: a seam node will created
                                capacity1 = self.global_nodes[node_nb_global].bottomCost

                                d3 = abs_distance(self.global_nodes[nodeNbGlobalBottom].colorOtherPatch,
                                                  self.croppedInputImage[j + 1, i])
                                d4 = abs_distance(self.croppedInputImage[j, i],
                                                  self.global_nodes[node_nb_global].colorOtherPatch)
                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (image_gy[j, i] / 255.0) \
                                       + (self.global_nodes[nodeNbGlobalBottom].gradYOtherPatch / 255.0)
                                grad += 1.0
                                capacity2 = (d1 + d3) / grad

                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (image_gy[j + 1, i] / 255.0) \
                                       + (self.global_nodes[node_nb_global].gradYOtherPatch / 255.0)
                                grad += 1.0
                                capacity3 = (d4 + d2) / grad

                                capacity2 += minCap
                                capacity3 += minCap
                                self.seamNode.append(
                                    SeamNode(node_nb_local, nodeNbLocalBottom, capacity1, capacity2, capacity3, 1))
                            else:
                                # No old seam
                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (image_gx[j, i] / 255.0) \
                                       + (image_gx[j + 1, i] / 255.0)
                                grad += 1.0
                                capBottom = (d1 + d2) / grad
                                capBottom += minCap
                                g.add_edge(node_nb_local, nodeNbLocalBottom, capBottom, capBottom)
                                self.global_nodes[node_nb_global].bottomCost = capBottom

                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(node_nb_local, nodeNbLocalBottom, 0.0, 0.0)
                            self.global_nodes[node_nb_global].bottomCost = 0.0
                    else:
                        # No overlap
                        g.add_edge(node_nb_local, nodeNbLocalBottom, 0.0, 0.0)
                        self.global_nodes[node_nb_global].bottomCost = 0.0

        return overlap

    def insert_patch(self, y: int, x: int, filling=True, blending=False, radius=0, err_p1_y=0, err_p1_x=0, err_p2_y=0,
                     err_p2_x=0,
                     random_refine=False):
        # y: new patch offset in global space, can be < 0

        node_nb_global: int = 0

        # v1: left top point in input image space
        v1_x: int = max(0, x) - x
        v1_y: int = max(0, y) - y
        self.cropped_left_top_input_space = (v1_y, v1_x)

        # v2: right bottom point in input image space
        v2_x: int = min(self.output_width - 1, x + self.input_width - 1) - x
        v2_y: int = min(self.output_height - 1, y + self.input_height - 1) - y
        self.cropped_right_bottom_input_space = (v2_y, v2_x)
        self.cropped_left_top_output_space = (y + v1_y, x + v1_x)
        self.cropped_right_bottom_output_space = (y + v2_y, x + v1_x)

        self.cropped_input_w = v2_x - v1_x + 1
        self.cropped_input_h = v2_y - v1_y + 1

        # print('Cropped patch size: {} * {}'.format(self.croppedInput_w, self.croppedInput_h))

        if self.cropped_input_w == 0 or self.cropped_input_h == 0:
            print("Patch lies outside output texture")
            return -1

        self.croppedInputImage = self.input_img[v1_y:v2_y + 1, v1_x: v2_x + 1]
        self.croppedInputImageGY = self.inputImageGY[v1_y:v2_y + 1, v1_x: v2_x + 1]
        self.croppedInputImageGX = self.inputImageGX[v1_y:v2_y + 1, v1_x: v2_x + 1]

        # Update origin coordinates, now in "real" global space
        x = max(0, x)
        y = max(0, y)

        self.linked_to_sink = set()
        self.linked_to_source = set()

        no_overlap = self.count_non_overlap(x, y)

        if no_overlap == self.cropped_input_w * self.cropped_input_h:
            print("No overlap detected")
            self.direct_copy_input_to_output(y=y, x=x)
            return 0
        elif no_overlap == 0 and filling:
            print('Patch does not contribute in filling')
            return -1

        image = self.compute_cropped_output_image(y=y, x=x)

        imageGY, imageGX = compute_gradient_image(image)

        # Graph construction
        nb_nodes = self.cropped_input_w * self.cropped_input_h
        nb_edges = (self.cropped_input_w - 1) * (self.cropped_input_h - 1) * 2 + (self.cropped_input_w - 1) + (
                self.cropped_input_h - 1)
        nb_seam_nodes = nb_edges
        nb_edges += nb_seam_nodes

        # allocate memory
        g = maxflow.Graph[float](nb_nodes + 2 * nb_seam_nodes, nb_edges)

        g.add_nodes(nb_nodes)

        overlap = self.configure_node_capacity(y, x, g=g, image_gy=imageGY, image_gx=imageGX)

        # print("Number of seam nodes: {}".format(len(self.seamNode)))
        node_old_seams = g.add_nodes(len(self.seamNode))

        self.num_sink = 0

        self.add_seam_node(g=g, node_ids=node_old_seams)

        # Assignments to source node
        self.link_source_node(y=y, x=x, g=g)
        self.link_sink_node(y, x, g, filling,
                            random_refine=random_refine,
                            radius=radius,
                            err_p1_y=err_p1_y, err_p1_x=err_p1_x, err_p2_y=err_p2_y, err_p2_x=err_p2_x)

        flow = g.maxflow()

        self.process_seams(y, x, g)

        self.merge_pixels(y, x, g, imageGY, imageGX, blending)

        self.seamNode = []

        return overlap

    def merge_pixels(self, y, x, g, imageGY, imageGX, blending):
        if blending:
            pass
        else:
            for j in range(self.cropped_input_h):
                for i in range(self.cropped_input_w):
                    node_nb_global = self.get_node_number_global(x, y, i, j)
                    the_node = self.global_nodes[node_nb_global]
                    if self.global_nodes[node_nb_global].empty:
                        # New pixel insertion
                        self.global_nodes[node_nb_global].color = self.croppedInputImage[j, i]
                        self.global_nodes[node_nb_global].empty = False
                    else:
                        if g.get_segment(self.get_node_number_local(i, j)) == source_type:
                            the_node.colorOtherPatch = self.croppedInputImage[j, i]
                            the_node.gradXOtherPatch = self.croppedInputImageGX[j, i]
                            the_node.gradYOtherPatch = self.croppedInputImageGY[j, i]

                        else:
                            the_node.colorOtherPatch = the_node.color
                            the_node.gradXOtherPatch = imageGX[j, i]
                            the_node.gradYOtherPatch = imageGY[j, i]
                            the_node.color = self.croppedInputImage[j, i]
                            the_node.empty = False

                            if not the_node.newSeam:
                                the_node.seamRight = False
                                the_node.seamBottom = False

                    the_node.newSeam = False

    def process_seams(self, y, x, g):
        k = 0
        for j in range(self.cropped_input_h):
            for i in range(self.cropped_input_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                node_nb_local = self.get_node_number_local(i, j)
                the_node = self.global_nodes[node_nb_global]
                if i < self.cropped_input_w - 1:
                    if g.get_segment(node_nb_local) != g.get_segment(self.get_node_number_local(i + 1, j)):
                        the_node.newSeam = True

                if j < self.cropped_input_h - 1:
                    if g.get_segment(node_nb_local) != g.get_segment(self.get_node_number_local(i, j + 1)):
                        the_node.newSeam = True

                if len(self.seamNode) and (k < len(self.seamNode)) and (node_nb_local == self.seamNode[k].start):
                    # Process old seam
                    currentSeamNode = self.seamNode[k].seam
                    currentSeamNodeEnd = self.seamNode[k].end

                    if g.get_segment(node_nb_local) == source_type and g.get_segment(currentSeamNodeEnd) == sink_type:
                        # Old seam remains with new seam cost
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.global_nodes[node_nb_global].rightCost = self.seamNode[k].capacity3
                                self.global_nodes[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.global_nodes[node_nb_global].bottomCost = self.seamNode[k].capacity3
                                self.global_nodes[node_nb_global].seamBottom = True

                        else:
                            # Old seam remains with new seam cost
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.global_nodes[node_nb_global].rightCost = self.seamNode[k].capacity2
                                self.global_nodes[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.global_nodes[node_nb_global].bottomCost = self.seamNode[k].capacity2
                                self.global_nodes[node_nb_global].seamBottom = True
                    elif g.get_segment(node_nb_local) == sink_type and g.get_segment(currentSeamNodeEnd) == source_type:
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.global_nodes[node_nb_global].rightCost = self.seamNode[k].capacity2
                                self.global_nodes[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.global_nodes[node_nb_global].bottomCost = self.seamNode[k].capacity2
                                self.global_nodes[node_nb_global].seamBottom = True
                        else:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.global_nodes[node_nb_global].rightCost = self.seamNode[k].capacity3
                                self.global_nodes[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.global_nodes[node_nb_global].bottomCost = self.seamNode[k].capacity3
                                self.global_nodes[node_nb_global].seamBottom = True
                    elif g.get_segment(currentSeamNode) == source_type:
                        if self.seamNode[k].orientation == 0:
                            self.global_nodes[node_nb_global].rightCost = self.seamNode[k].capacity1
                            self.global_nodes[node_nb_global].seamRight = True
                        else:
                            self.global_nodes[node_nb_global].bottomCost = self.seamNode[k].capacity1
                            self.global_nodes[node_nb_global].seamBottom = True

                    else:
                        pass

                    k += 1
                else:
                    # New seam
                    if i < self.cropped_input_w - 1:
                        if g.get_segment(node_nb_local) != g.get_segment(self.get_node_number_local(i + 1, j)):
                            self.global_nodes[node_nb_global].seamRight = True

                    if j < self.cropped_input_h - 1:
                        if g.get_segment(node_nb_local) != g.get_segment(self.get_node_number_local(i, j + 1)):
                            self.global_nodes[node_nb_global].seamBottom = True

    def add_seam_node(self, g, node_ids):
        for i in range(len(self.seamNode)):
            node_old_seam = node_ids[i]
            self.seamNode[i].seam = node_old_seam
            g.add_edge(self.seamNode[i].start, node_old_seam, self.seamNode[i].capacity2, self.seamNode[i].capacity2)
            g.add_edge(node_old_seam, self.seamNode[i].end, self.seamNode[i].capacity3, self.seamNode[i].capacity3)
            g.add_tedge(node_old_seam, 0.0, self.seamNode[i].capacity1)
            self.num_sink += 1

    def link_source_node(self, y, x, g):
        for i in range(self.cropped_input_w):
            j = 0
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.global_nodes[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)
                self.linked_to_source.add(self.get_node_number_local(i, j))

            j = self.cropped_input_h - 1
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.global_nodes[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)
                self.linked_to_source.add(self.get_node_number_local(i, j))

        for j in range(self.cropped_input_h):
            i = 0
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.global_nodes[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)
                self.linked_to_source.add(self.get_node_number_local(i, j))

            i = self.cropped_input_w - 1
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.global_nodes[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)
                self.linked_to_source.add(self.get_node_number_local(i, j))

    def link_sink_node(self, y, x, g, filling, random_refine, radius, err_p1_y, err_p1_x, err_p2_y, err_p2_x):
        if filling:
            print('Filling patch ', self.patch_number)
            for j in range(self.cropped_input_h):
                for i in range(self.cropped_input_w):
                    node_nb_global = self.get_node_number_global(x, y, i, j)
                    if not self.global_nodes[node_nb_global].empty:
                        nodeNbGlobalLeft = self.get_node_number_global(x, y, i - 1, j)
                        nodeNbGlobalRight = self.get_node_number_global(x, y, i + 1, j)
                        nodeNbGlobalTop = self.get_node_number_global(x, y, i, j - 1)
                        nodeNbGlobalBottom = self.get_node_number_global(x, y, i, j + 1)

                        # empty left, right, top, bottom
                        if (nodeNbGlobalLeft != -1) and self.global_nodes[nodeNbGlobalLeft].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalTop != -1) and self.global_nodes[nodeNbGlobalTop].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalRight != -1) and self.global_nodes[nodeNbGlobalRight].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalBottom != -1) and self.global_nodes[nodeNbGlobalBottom].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                    else:
                        # this pixel is not filled, will be first filled by this patch, assign to sink
                        g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                        self.num_sink += 1
        else:
            # Refinement
            print('Refinement patch ', self.patch_number)
            if not random_refine:
                # outputX = inputX + tX
                # outputY = inputY + tY
                # v1_x = max(0, outputX)
                # v1_y = max(0, outputY)
                # v2_x = min(self.output_width - 1, outputX + radius - 1)
                # v2_y = min(self.output_height - 1, outputY + radius - 1)
                # # Subpatch origin coordinates in output space
                # outputX = v1_x
                # outputY = v1_y
                # # Subpatch origin coordinates in input space
                # inputX = outputX - tX
                # inputY = outputY - tY
                #
                # cj = 0
                # ci = 0
                err_node = 0
                for j in range(err_p1_y, err_p2_y + 1):
                    for i in range(err_p1_x, err_p2_x + 1):
                        j_local = j - self.cropped_left_top_output_space[0]
                        i_local = i - self.cropped_left_top_output_space[1]

                        node_nb_local = self.get_node_number_local(i_local, j_local)
                        if node_nb_local not in self.linked_to_source:
                            g.add_tedge(node_nb_local, 0.0, infiniteCap)
                        else:
                            # print('linked to source')
                            pass
                        self.num_sink += 1
                        err_node += 1
                # print('error node', err_node)
            else:
                # random refinement
                if self.num_sink == 0:
                    # at least one pixel from new patch should be added
                    # so add the center pixel to sink
                    # this should not happen very often
                    j = self.cropped_input_h // 2
                    i = self.cropped_input_w // 2
                    node_nb_local = self.get_node_number_local(i, j)
                    # node_nb_global = self.getNodeNbGlobal(x, y, i, j)
                    g.add_tedge(node_nb_local, 0.0, infiniteCap)
                    self.num_sink += 1
                else:
                    pass
                    # print('already connected')
            # print('sink number', self.num_sink)
        assert self.num_sink > 0

    # -------------------------------- START: helper function --------------------------------

    def get_node_number_local(self, i: int, j: int) -> int:
        return i + self.cropped_input_w * j

    def get_node_number_global(self, x, y, i, j) -> int:
        if (x + i < self.output_width) and (x + i >= 0) and (y + j < self.output_height) and (y + j >= 0):
            return (x + i) + self.output_width * (y + j)
        else:
            return -1

    def get_position_global(self, node_nb_global):
        x = node_nb_global % self.output_width
        y = node_nb_global // self.output_width
        return y, x

    def add_img_border(self, img, border_top, border_bottom, border_left, border_right):
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_bordered = np.zeros((img_h + border_top + border_bottom, img_w + border_left + border_right, 3),
                                dtype=int)
        img_bordered[border_top: border_top + img_h, border_left: border_left + img_w] = img
        return img_bordered

    def add_mask_border(self, mask, border_top, border_bottom, border_left, border_right):
        mask_h = mask.shape[0]
        mask_w = mask.shape[1]
        mask_bordered = np.zeros((mask_h + border_top + border_bottom, mask_w + border_left + border_right,),
                                 dtype=int)
        mask_bordered[border_top: border_top + mask_h, border_left: border_left + mask_w] = mask
        return mask_bordered

    # -------------------------------- END: helper function --------------------------------

    # -------------------------------- START: entire patch matching placement algorithm --------------------------------
    def entire_matching_filling(self, k, blending=False, save_img=False, show_seams=False):
        print('Initial synthesis: Entire Patch matching')
        window_width = self.input_width // 4
        window_height = self.input_height // 4

        offset_y = 0

        used_cost = np.zeros((self.output_height, self.output_width), dtype=bool)
        # # y is sample in (offset_y-overlap_height, offset_y]
        # y = offset_y - (window_height + randint(0, window_height - 1))

        # y = randint(offset_y - self.input_height + window_height, offset_y - 1)
        while offset_y < self.output_height:  # loop y
            print('New Row')
            offset_x = 0
            # # sample x in [-overlap_width, 0)
            # x = offset_x - (window_width + randint(0, window_width - 1))
            # x = randint(offset_x - self.input_width + window_width, offset_x - 1)
            while offset_x < self.output_width:  # loop x
                self.write_image()

                window_filled_mask = self.output_img_filled_mask[offset_y:offset_y + window_height,
                                     offset_x:offset_x + window_width]
                if window_filled_mask.all():
                    # this window is filled, skip
                    offset_x += window_width
                    continue
                # calculate cost matrix
                cost_matrix = self.compute_entire_matching_cost_matrix(
                    y_start=offset_y - self.input_height + window_height,
                    y_end=offset_y,
                    x_start=offset_x - self.input_width + window_width,
                    x_end=offset_x)

                # sample y, x according to the cost matrix

                probs = np.exp(-cost_matrix / self.sig2 / k)
                if probs.sum() == 0.0:
                    print('k is too small, all probs is zero')
                    probs[np.where(cost_matrix == np.min(cost_matrix))] = 1
                    probs = probs / probs.sum()  # all probabilities should add up to one

                else:
                    probs = probs / probs.sum()  # all probabilities should add up to one
                probs = probs.reshape((-1,))
                index = np.random.choice(probs.shape[0], p=probs)
                index_x = index % cost_matrix.shape[1]
                index_y = index // cost_matrix.shape[1]
                y = index_y + offset_y - self.input_height + window_height
                x = index_x + offset_x - self.input_width + window_width

                if y < self.output_height and x < self.output_width:
                    res = self.insert_patch(y, x)
                    if res != -1:  # some pixel is added
                        self.patch_number += 1
                        self.update_seams_max_error()

                        if save_img:
                            self.write_image()
                            self.save_output_img(self.patch_number)
                            if show_seams:
                                self.reveal_seams()
                                self.reveal_seams_max_error(2)
                                self.save_output_img(str(self.patch_number) + '_s')
                else:
                    print('Error')

                offset_x += window_width - window_width // 4  # go to next column

                # if offset_x > self.output_width:
                #     break
                # the x candidate is in [offset_x-input_width+window_width, offset_x)
                # the y candidate is in [offset_y-input_height+window_height, offset_y)

                # min_cost = cost_matrix[index_y,index_x]
                # if x >= self.output_width:
                #     break

            offset_y += window_height - window_height // 4  # go to next row

            # if offset_y > self.output_height:
            #     break
            # # sample y
            # y = offset_y - (window_height + randint(0, window_height - 1))

            # if y >= self.output_height:
            #     break

    def entire_matching_refinement(self, iter=20, k=0.1, error_radius=2, save_img=False, show_seams=False):
        print('Refinement: Entire Patch matching')
        padding_height = self.input_height // 3
        padding_width = self.input_width // 3

        blending = False
        for j in range(iter):
            self.write_image()
            v1y, v1x, v2y, v2x = self.pick_error_region(error_radius)
            p1y = v2y - self.input_height
            p1x = v2x - self.input_width
            p2y = v1y
            p2x = v1x

            # calculate cost matrix
            cost_matrix = self.compute_entire_matching_cost_matrix(
                y_start=p1y,
                y_end=p2y,
                x_start=p1x,
                x_end=p2x)

            # sample y, x according to the cost matrix

            probs = np.exp(-cost_matrix / self.sig2 / k)
            if probs.sum() == 0.0:
                print('k is too small, all probs is zero')
                probs[np.where(cost_matrix == np.min(cost_matrix))] = 1
                probs = probs / probs.sum()  # all probabilities should add up to one

            else:
                probs = probs / probs.sum()  # all probabilities should add up to one
            probs = probs.reshape((-1,))
            index = np.random.choice(probs.shape[0], p=probs)
            index_x = index % cost_matrix.shape[1]
            index_y = index // cost_matrix.shape[1]
            y = index_y + p1y
            x = index_x + p1x

            if y < self.output_height and x < self.output_width:
                res = self.insert_patch(y, x, filling=False, radius=error_radius, random_refine=False,
                                        err_p1_y=v1y,
                                        err_p1_x=v1x,
                                        err_p2_y=v2y,
                                        err_p2_x=v2x)
                if res != -1:  # some pixel is added
                    self.patch_number += 1
                    self.update_seams_max_error(radius=error_radius)

                    if save_img:
                        self.write_image()
                        self.save_output_img(self.patch_number)
                        if show_seams:
                            self.reveal_seams()
                            self.reveal_seams_max_error(error_radius)
                            self.save_output_img(str(self.patch_number) + '_s')
            else:
                print('Error')

    # -------------------------------- END: entire patch matching placement algorithm --------------------------------

    # -------------------------------- START: sub patch matching placement algorithm --------------------------------
    def sub_matching_filling(self, k, blending=False, save_img=False, show_seams=False):
        return self.entire_matching_filling(k, blending, save_img, show_seams)

    def sub_matching_refinement(self, iter=20, k=0.1, error_radius=2, save_img=False, show_seams=False):
        print('Refinement: Sub Patch matching')
        self.used_offset = set()
        for j in range(iter):
            self.write_image()
            v1y, v1x, v2y, v2x = self.pick_error_region(error_radius)
            # p1y = v2y - self.input_height
            # p1x = v2x - self.input_width
            # p2y = v1y
            # p2x = v1x

            # calculate cost matrix
            # cost_matrix = self.compute_sub_matching_cost_matrix(
            #     y_start=v1y,
            #     y_end=v2y,
            #     x_start=v1x,
            #     x_end=v2x)

            cost_matrix = self.compute_sub_matching_cost_matrix_fft(
                y_start=v1y,
                y_end=v2y,
                x_start=v1x,
                x_end=v2x)

            while True:
                probs = np.exp(-cost_matrix / self.sig2 / k)
                if probs.sum() == 0.0:
                    print('k is too small, all probs is zero')
                    probs[np.where(cost_matrix == np.min(cost_matrix))] = 1
                    probs = probs / probs.sum()  # all probabilities should add up to one

                else:
                    probs = probs / probs.sum()  # all probabilities should add up to one
                probs = probs.reshape((-1,))
                index = np.random.choice(probs.shape[0], p=probs)
                index_x = index % cost_matrix.shape[1]
                index_y = index // cost_matrix.shape[1]

                y = v1y - index_y
                x = v1x - index_x

                if (y, x) in self.used_offset:
                    cost_matrix[index_y, index_x] = 10000
                    # sample again
                else:
                    break

            if y < self.output_height and x < self.output_width:
                res = self.insert_patch(y, x, filling=False, radius=error_radius, random_refine=False,
                                        err_p1_y=v1y,
                                        err_p1_x=v1x,
                                        err_p2_y=v2y,
                                        err_p2_x=v2x)
                if res != -1:  # some pixel is added
                    self.used_offset.add((y, x))
                    self.patch_number += 1
                    self.update_seams_max_error(radius=error_radius)

                    if save_img:
                        self.write_image()
                        self.save_output_img(self.patch_number)
                        if show_seams:
                            self.reveal_seams()
                            self.reveal_seams_max_error(error_radius)
                            self.save_output_img(str(self.patch_number) + '_s')
            else:
                print('Error')

    # -------------------------------- END: sub patch matching placement algorithm --------------------------------

    # -------------------------------- START: random placement algorithm --------------------------------

    def random_fill(self, save_img=False, show_seams=False):
        print('Initial synthesis: Random')
        window_width = self.input_width // 5
        window_height = self.input_height // 5

        offset_y = 0

        # y is sample in (offset_y-overlap_height, offset_y]
        # y = offset_y - (overlap_height + randint(0, overlap_height - 1))

        while offset_y < self.output_height:  # loop y
            print('New Row')
            offset_x = 0
            # sample x in [-overlap_width, 0)
            # x = offset_x - (overlap_width + randint(0, overlap_width - 1))
            while offset_x < self.output_width:  # loop x
                self.write_image()

                window_filled_mask = self.output_img_filled_mask[offset_y:offset_y + window_height,
                                     offset_x:offset_x + window_width]
                if window_filled_mask.all():
                    # this window is filled, skip
                    offset_x += window_width
                    continue

                y = randint(offset_y - self.input_height + window_height, offset_y)
                x = randint(offset_x - self.input_width + window_width, offset_x)

                if y < self.output_height and x < self.output_width:
                    res = self.insert_patch(y, x)
                    if res != -1:  # some pixel is added
                        self.patch_number += 1
                        self.update_seams_max_error()

                        if save_img:
                            self.write_image()
                            self.save_output_img(self.patch_number)
                            if show_seams:
                                self.reveal_seams()
                                self.reveal_seams_max_error(2)
                                self.save_output_img(str(self.patch_number) + '_s')
                else:
                    print('Error')

                offset_x += window_width  # go to next column

            offset_y += window_height  # go to next row

    def random_refinement(self, iter=20, error_radius=2, save_img=False, show_seams=False):
        print('Refinement: Random')
        window_width = self.input_width // 3
        window_height = self.input_height // 3
        blending = False
        for k in range(iter):
            self.write_image()
            v1y, v1x, v2y, v2x = self.pick_error_region(error_radius)

            # sample y and x
            x = randint(v2x - self.input_width, v1x)
            y = randint(v2y - self.input_height, v1y)
            self.insert_patch(y, x, filling=False, blending=blending, random_refine=False, radius=error_radius,
                              err_p1_y=v1y,
                              err_p1_x=v1x,
                              err_p2_y=v2y,
                              err_p2_x=v2x
                              )
            self.update_seams_max_error()
            self.patch_number += 1
            if save_img:
                self.write_image()
                self.save_output_img(self.patch_number)
                if show_seams:
                    self.reveal_seams()
                    self.reveal_seams_max_error(error_radius)
                    self.save_output_img(str(self.patch_number) + '_s')

    # -------------------------------- END: random placement algorithm --------------------------------

    # -------------------------------- START: image plot related functions --------------------------------
    def write_image(self):
        for j in range(self.output_height):
            for i in range(self.output_width):
                the_node = self.global_nodes[self.get_node_number_global(0, 0, i, j)]
                if not the_node.empty:
                    self.output_img[j, i] = the_node.color
                    self.output_img_filled_mask[j, i] = 1
                else:
                    self.output_img[j, i] = [0, 0, 0]
                    self.output_img_filled_mask[j, i] = 0

    def show_output_img(self):
        plt.subplot(1, 1, 1)
        plt.imshow(self.output_img)
        plt.show()

    def save_output_img(self, patch_id):
        file_basename = os.path.basename(data_filename)
        file_basename_no_ext = os.path.splitext(file_basename)[0]
        name = os.path.join(out_dir, '{}_{}.png'.format(file_basename_no_ext, patch_id))
        plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.output_img)
        plt.savefig(name)

    def reveal_seams(self):
        for j in range(self.output_height):
            for i in range(self.output_width):
                node_nb_global = self.get_node_number_global(0, 0, i, j)
                the_node = self.global_nodes[node_nb_global]
                if not the_node.empty:
                    if the_node.seamRight or the_node.seamBottom:
                        for sj in range(-0, 1):
                            for si in range(-0, 1):
                                tj = j + sj
                                ti = i + si
                                if (0 <= ti < self.output_width) and (0 <= tj < self.output_height):
                                    self.output_img[tj, ti] = [255, 0, 0]

    def reveal_seams_max_error(self, radius):
        """
        draw a rectangle
        :param radius: rect width = 2*raduis+1
        :return:
        """
        if self.maxErrNodeNbGlobal == -1:
            return
        max_err_y, max_err_x = self.get_position_global(self.maxErrNodeNbGlobal)
        for sj in range(-radius, radius + 1):
            for si in range(-radius, radius + 1):
                tj = max_err_y + sj
                ti = max_err_x + si
                if 0 <= ti <= self.output_width and tj >= 0 and tj <= self.output_height:
                    self.output_img[tj, ti] = [255, 0, 255]

    # -------------------------------- END: image plot related functions --------------------------------

    def update_seams_max_error(self, radius=0):
        """
        find a node with max seam error nearby
        used for picking error region
        """
        self.maxErrNodeNbGlobal = -1
        maxErr = -1.0
        bs = self.borderSize
        for j in range(bs, self.output_height - bs):
            for i in range(bs, self.output_width - bs):
                err_sum = 0.0
                nodeNbGlobal = self.get_node_number_global(0, 0, i, j)

                # all the error nearby
                for jj in range(-radius, radius + 1):
                    for ii in range(-radius, radius + 1):
                        node_neighbor_global = self.get_node_number_global(0, 0, i + ii, j + jj)
                        neighbor_node = self.global_nodes[node_neighbor_global]
                        if not neighbor_node.empty:
                            if neighbor_node.seamRight:
                                err_sum += neighbor_node.rightCost
                            if neighbor_node.seamBottom:
                                err_sum += neighbor_node.bottomCost
                if err_sum > maxErr:
                    maxErr = err_sum
                    self.maxErrNodeNbGlobal = nodeNbGlobal

                # nodeNbGlobal = self.get_node_number_global(0, 0, i, j)
                # the_node = self.global_nodes[nodeNbGlobal]
                # if not the_node.empty:
                #     if the_node.seamRight:
                #         if the_node.rightCost > maxErr:
                #             maxErr = the_node.rightCost
                #             self.maxErrNodeNbGlobal = nodeNbGlobal
                #
                #     if the_node.seamBottom:
                #         if the_node.bottomCost > maxErr:
                #             maxErr = the_node.bottomCost
                #             self.maxErrNodeNbGlobal = nodeNbGlobal

        return maxErr

    def compute_sub_matching_cost_matrix(self, y_start, y_end, x_start, x_end):
        # search the error region inside input image
        error_region_cropped = self.output_img[y_start:y_end + 1, x_start:x_end + 1]
        cropped_out_mask = self.output_img_filled_mask[y_start:y_end + 1, x_start:x_end + 1]
        overlap_pixel_count = cropped_out_mask.sum()

        err_region_width = x_end - x_start + 1
        err_region_height = y_end - y_start + 1
        possible_translation_height = self.input_height - err_region_height + 1
        possible_translation_width = self.input_width - err_region_width + 1
        cost_matrix = np.zeros((possible_translation_height, possible_translation_width),
                               dtype=float)

        cost_j = 0
        for j in range(possible_translation_height):
            cost_i = 0
            for i in range(possible_translation_width):
                sub_patch_input = self.input_img[j:j + err_region_height, i:i + err_region_width]
                distance = (sub_patch_input - error_region_cropped).sum(axis=2) / 3 * cropped_out_mask
                distance = distance / 255.0
                distance_squared = distance * distance

                if overlap_pixel_count > 0:
                    cost = distance_squared.sum() / overlap_pixel_count
                    cost_matrix[cost_j, cost_i] = cost
                else:
                    cost_matrix[cost_j, cost_i] = 0

                cost_i += 1
            cost_j += 1

        return cost_matrix

    def compute_sub_matching_cost_matrix_fft(self, y_start, y_end, x_start, x_end):
        # search the error region inside input image
        error_region_cropped = self.output_img[y_start:y_end + 1, x_start:x_end + 1]
        # cropped_out_mask = self.output_img_filled_mask[y_start:y_end + 1, x_start:x_end + 1]
        # overlap_pixel_count = cropped_out_mask.sum()
        overlap_pixel_count = (y_end + 1 - y_start) * (x_end + 1 - x_start)

        err_region_width = x_end - x_start + 1
        err_region_height = y_end - y_start + 1

        possible_translation_height = self.input_height - err_region_height + 1
        possible_translation_width = self.input_width - err_region_width + 1

        # cost_matrix = np.zeros((possible_translation_height, possible_translation_width),
        #                        dtype=float)
        #
        # tmp = error_region_cropped- self.input_img[:err_region_height, : err_region_width]
        # tmp = tmp/255.0
        # tmp = tmp.sum(axis=2) / 3
        # tmp = tmp*tmp
        # tmp = tmp.sum() / overlap_pixel_count

        # assert cropped_out_mask.all()

        input_avg = self.input_img.sum(axis=2) / 3 / 255.0
        err_avg = error_region_cropped.sum(axis=2) / 3 / 255.0
        # i_2 = (self.input_img/255.0)*(self.input_img/255.0)
        i_2 = input_avg * input_avg
        term_1 = signal.fftconvolve(i_2, np.ones_like(err_avg), mode='valid')
        term_1 = term_1.reshape(term_1.shape[0], term_1.shape[1])
        #
        # term1 = np.roll(self.summed_area_table_i_squared, shift=(-err_region_height, -err_region_width), axis=(0, 1)) \
        #         - np.roll(self.summed_area_table_i_squared, shift=-err_region_height, axis=0) \
        #         - np.roll(self.summed_area_table_i_squared, shift=-err_region_width, axis=1) \
        #         + self.summed_area_table_i_squared
        # term1 = term1[:possible_translation_height, :possible_translation_width]
        # print(term1 - term_1)
        term_2 = (error_region_cropped / 255.0).sum(axis=2) / 3
        term_2 = term_2 * term_2
        term_2 = term_2.sum()

        cost_conv = signal.fftconvolve(input_avg, np.flip(err_avg, axis=(0, 1)), mode='valid')
        cost_conv = cost_conv.reshape(cost_conv.shape[0], cost_conv.shape[1])

        cost_all = term_1 + term_2 - 2 * cost_conv
        cost_all = cost_all / overlap_pixel_count

        # cost_j = 0
        # for j in range(self.input_height - err_region_height+1):
        #     cost_i = 0
        #     for i in range(self.input_width - err_region_width+1):
        #         sub_patch_input = self.input_img[j:j + err_region_height, i:i + err_region_width]
        #         distance = (sub_patch_input - error_region_cropped).sum(axis=2) / 3
        #         distance = distance / 255.0
        #         distance_squared = distance * distance
        #
        #         if overlap_pixel_count > 0:
        #             cost = distance_squared.sum() / overlap_pixel_count
        #             cost_matrix[cost_j, cost_i] = cost
        #         else:
        #             cost_matrix[cost_j, cost_i] = 0
        #
        #         cost_i += 1
        #     cost_j += 1
        #
        # plt.subplot(1,2,1)
        #
        # fig, (ax_cost_fft, ax_cost, ax_diff) = plt.subplots(3, 1,
        #                                                      figsize=(6, 15))
        # ax_cost_fft.imshow(cost_all, cmap='gray')
        # ax_cost_fft.set_title('Cost matrix by FFT')
        # # ax_cost_fft.set_axis_off()
        # ax_cost.imshow(cost_matrix, cmap='gray')
        # ax_cost.set_title('Cost matrix by naive search')
        # # ax_cost.set_axis_off()
        # im =ax_diff.imshow(np.abs(cost_all-cost_matrix), cmap='gray')
        # ax_diff.set_title('Cost matrix difference')
        # fig.colorbar(im)
        # # ax_diff.set_axis_off()
        # fig.show()
        # # print('max',cost_matrix.max(), cost_all.max())
        #
        # # print('min',cost_matrix.min(), cost_all.min())
        # print(tmp, cost_matrix[0,0], cost_all[0,0])
        return cost_all

    def compute_entire_matching_cost_matrix(self, y_start, y_end, x_start, x_end):
        matrix_height = y_end - y_start + 1
        matrix_width = x_end - x_start + 1
        output_img_padded = self.add_img_border(self.output_img, matrix_height, self.input_height, matrix_width,
                                                self.input_width)
        output_mask_padded = self.add_mask_border(self.output_img_filled_mask, matrix_height, self.input_height,
                                                  matrix_width, self.input_width)

        # convert from global space to padded global space
        y_start += matrix_height
        y_end += matrix_height
        x_start += matrix_width
        x_end += matrix_width

        cost_matrix = np.zeros((matrix_height, matrix_width), dtype=float)
        cost_j = 0
        for translate_y in range(y_start, y_end + 1):
            cost_i = 0
            for translate_x in range(x_start, x_end + 1):
                cropped_out_img = output_img_padded[translate_y:translate_y + self.input_height,
                                  translate_x:translate_x + self.input_width]
                cropped_out_mask = output_mask_padded[translate_y:translate_y + self.input_height,
                                   translate_x:translate_x + self.input_width]
                overlap_pixel_count = cropped_out_mask.sum()
                distance = (self.input_img - cropped_out_img).sum(axis=2) / 3 * cropped_out_mask
                distance = distance / 255.0
                distance_squared = distance * distance

                if overlap_pixel_count > 0:
                    cost = distance_squared.sum() / overlap_pixel_count
                    cost_matrix[cost_j, cost_i] = cost
                else:
                    cost_matrix[cost_j, cost_i] = 0

                cost_i += 1
            cost_j += 1

        return cost_matrix


def test():
    img_in = imread('data/strawberries2.gif')
    if img_in.shape[2] == 4:
        # remove alpha channel
        img_in = np.array(img_in[:, :, 0:3])

    global out_dir
    out_dir = 'out/graph_cut_texture'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    gc_texture = GraphCutTexture(img_in, img_in.shape[0], img_in.shape[1] * 2)
    gc_texture.insert_patch(0, 0)

    gc_texture.insert_patch(0, img_in.shape[1] - 100)
    gc_texture.write_image()
    gc_texture.save_output_img('test')

    gc_texture.reveal_seams()
    gc_texture.save_output_img('test-s')


# test()


if __name__ == "__main__":
    # data_filename = 'akeyboard_small.gif'
    # data_filename = 'strawberries2.gif'
    # data_filename = 'green.gif'
    # data_filename = 'jelly.gif'
    # data_filename = 'nuts6.gif'
    # data_filename = 'AB_valley.gif'
    # data_filename = 'AB_machu3.gif'
    data_filename = 'sheep.gif'


    input_file_path = os.path.join('data', data_filename)

    img_in = imread(input_file_path)
    if len(img_in.shape) == 3 and img_in.shape[2] == 4:
        # remove alpha channel
        img_in = np.array(img_in[:, :, 0:3])

    print('original image size: ', img_in.shape)

    gc_texture = GraphCutTexture(img_in, img_in.shape[0] * 2, img_in.shape[1] * 2)

    placement = 'random'
    if placement not in ['random', 'entire', 'sub']:
        print('unknown placement method', )
        exit(1)

    out_dir = os.path.join('out/graph_cut_texture', placement)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if placement == 'random':
        gc_texture.random_fill(save_img=True, show_seams=True)
        gc_texture.random_refinement(iter=40, error_radius=5, save_img=True, show_seams=True)
    elif placement == 'entire':
        gc_texture.entire_matching_filling(k=0.001, save_img=True, show_seams=True)
        gc_texture.entire_matching_refinement(iter=20, k=0.001, error_radius=8, save_img=True, show_seams=True)
    elif placement == 'sub':
        gc_texture.sub_matching_filling(k=0.01, save_img=True, show_seams=True)
        gc_texture.sub_matching_refinement(iter=40, k=0.01, error_radius=8, save_img=True, show_seams=True)

    gc_texture.write_image()
    # gc_texture.reveal_seams()
    gc_texture.save_output_img('final')
    # gc_texture.show_output_img()
