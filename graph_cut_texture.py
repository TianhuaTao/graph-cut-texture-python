from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

import maxflow
from random import randint
import scipy.ndimage as nd
import os

plt.figure(num=None, figsize=(40, 32), dpi=80, facecolor='w', edgecolor='k')

minCap = 1e-7
infiniteCap = 1e12
source_type = 0
sink_type = 1
out_dir = ''  # assigned in main


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def square_distance(c1, c2):
    diff = (c1 / 255.0 - c2 / 255.0)
    return (diff * diff).sum()


def abs_distance(c1, c2):
    return (np.abs(c1 - c2)).sum()


def compute_gradient_image(image):
    x_kernel = 1 / 3 * np.array([[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]])
    y_kernel = 1 / 3 * np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])
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
        self.croppedInput_w = 0
        self.croppedInput_h = 0
        self.croppedInputImageGX = None
        self.croppedInputImageGY = None

        self.output_height = output_height
        self.output_width = output_width

        self.output_img = np.zeros((output_height, output_width, 3), dtype=int)
        self.output_img_filled_mask = np.zeros((output_height, output_width), dtype=int)
        self.patch_number = 0

        self.globalNode = [GlobalNode() for _ in range(output_width * output_height)]
        self.seamNode: [SeamNode] = []

        self.inputImageGY, self.inputImageGX = compute_gradient_image(self.input_img)
        self.borderSize = 16

        self.maxErrNodeNbGlobal = -1
        self.num_sink = 0

    def reveal_seams(self):
        seam_size = 1
        for j in range(self.output_height):
            for i in range(self.output_width):
                node_nb_global = self.get_node_number_global(0, 0, i, j)
                the_node = self.globalNode[node_nb_global]
                if not the_node.empty:
                    if the_node.seamRight or the_node.seamBottom:
                        for sj in range(-seam_size, seam_size):
                            for si in range(-seam_size, seam_size):
                                tj = j + sj
                                ti = i + si
                                if (0 <= ti < self.output_width) and (0 <= tj < self.output_height):
                                    self.output_img[tj, ti] = [255, 0, 0]

    def count_non_overlap(self, x, y):
        no_overlap = 0
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                if self.globalNode[node_nb_global].empty:
                    no_overlap += 1
        return no_overlap

    def direct_copy_input_to_output(self, y, x):
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                the_node = self.globalNode[node_nb_global]
                if the_node.empty:
                    the_node.color = self.croppedInputImage[j, i]
                    the_node.empty = False

    def compute_cropped_output_image(self, y, x):
        image = np.zeros_like(self.croppedInputImage)
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                if not self.globalNode[node_nb_global].empty:
                    image[j, i] = self.globalNode[node_nb_global].color
                else:
                    image[j, i] = [0, 0, 0]
        return image

    def configure_node_capacity(self, y, x, g, image_gy, image_gx):
        overlap: int = 0
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                node_nb_local = self.get_node_number_local(i, j)

                if i < self.croppedInput_w - 1:
                    node_nb_global_right = self.get_node_number_global(x, y, i + 1, j)
                    node_nb_local_right = self.get_node_number_local(i + 1, j)

                    if not self.globalNode[node_nb_global].empty:
                        if not self.globalNode[node_nb_global_right].empty:  # right
                            d1 = abs_distance(self.globalNode[node_nb_global].color, self.croppedInputImage[j, i])
                            d2 = abs_distance(self.globalNode[node_nb_global_right].color,
                                              self.croppedInputImage[j, i + 1])
                            if self.globalNode[node_nb_global].seamRight:
                                # Old seam: a seam node will created
                                capacity1 = self.globalNode[node_nb_global].rightCost
                                d3 = abs_distance(self.globalNode[node_nb_global_right].colorOtherPatch,
                                                  self.croppedInputImage[j, i + 1])
                                d4 = abs_distance(self.croppedInputImage[j, i],
                                                  self.globalNode[node_nb_global].colorOtherPatch)
                                grad = ((self.croppedInputImageGX[j, i] / 255.0)
                                        + (self.croppedInputImageGX[j, i + 1] / 255.0)
                                        + (image_gx[j, i] / 255.0)
                                        + (self.globalNode[node_nb_global_right].gradXOtherPatch / 255.0))
                                grad += 1.0

                                capacity2 = (d1 + d3) / grad
                                grad = (self.croppedInputImageGX[j, i] / 255.0) \
                                       + (self.croppedInputImageGX[j, i + 1] / 255.0) \
                                       + (image_gx[j, i + 1] / 255.0) \
                                       + (self.globalNode[node_nb_global].gradXOtherPatch / 255.0)
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
                                self.globalNode[node_nb_global].rightCost = capRight
                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(node_nb_local, node_nb_local_right, 0.0, 0.0)
                            self.globalNode[node_nb_global].rightCost = 0.0
                    else:
                        g.add_edge(node_nb_local, node_nb_local_right, 0.0, 0.0)
                        self.globalNode[node_nb_global].rightCost = 0.0

                if (j < self.croppedInput_h - 1):
                    nodeNbGlobalBottom = self.get_node_number_global(x, y, i, j + 1)
                    nodeNbLocalBottom = self.get_node_number_local(i, j + 1)

                    if not self.globalNode[node_nb_global].empty:
                        if not self.globalNode[nodeNbGlobalBottom].empty:  # bottom
                            # Overlap
                            d1 = abs_distance(self.globalNode[node_nb_global].color, self.croppedInputImage[j, i])
                            d2 = abs_distance(self.globalNode[nodeNbGlobalBottom].color,
                                              self.croppedInputImage[j + 1, i])
                            if self.globalNode[node_nb_global].seamBottom:
                                # Old seam: a seam node will created
                                capacity1 = self.globalNode[node_nb_global].bottomCost

                                d3 = abs_distance(self.globalNode[nodeNbGlobalBottom].colorOtherPatch,
                                                  self.croppedInputImage[j + 1, i])
                                d4 = abs_distance(self.croppedInputImage[j, i],
                                                  self.globalNode[node_nb_global].colorOtherPatch)
                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (image_gy[j, i] / 255.0) \
                                       + (self.globalNode[nodeNbGlobalBottom].gradYOtherPatch / 255.0)
                                grad += 1.0
                                capacity2 = (d1 + d3) / grad

                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (image_gy[j + 1, i] / 255.0) \
                                       + (self.globalNode[node_nb_global].gradYOtherPatch / 255.0)
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
                                self.globalNode[node_nb_global].bottomCost = capBottom

                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(node_nb_local, nodeNbLocalBottom, 0.0, 0.0)
                            self.globalNode[node_nb_global].bottomCost = 0.0
                    else:
                        # No overlap
                        g.add_edge(node_nb_local, nodeNbLocalBottom, 0.0, 0.0)
                        self.globalNode[node_nb_global].bottomCost = 0.0

        return overlap

    def insert_patch(self, y: int, x: int, filling=True, blending=False, radius=0, inputY=0, inputX=0, tY=0, tX=0,
                     random_refine=False):
        # y: new patch offset in global space, can be < 0

        node_nb_global: int = 0

        # v1: left top point in input image space
        v1_x: int = max(0, x) - x
        v1_y: int = max(0, y) - y

        # v2: right bottom point in input image space
        v2_x: int = min(self.output_width - 1, x + self.input_width - 1) - x
        v2_y: int = min(self.output_height - 1, y + self.input_height - 1) - y

        self.croppedInput_w = v2_x - v1_x + 1
        self.croppedInput_h = v2_y - v1_y + 1

        # print('Cropped patch size: {} * {}'.format(self.croppedInput_w, self.croppedInput_h))

        if self.croppedInput_w == 0 or self.croppedInput_h == 0:
            print("Patch lies outside output texture")
            return -1

        self.croppedInputImage = self.input_img[v1_y:v2_y + 1, v1_x: v2_x + 1]
        self.croppedInputImageGY = self.inputImageGY[v1_y:v2_y + 1, v1_x: v2_x + 1]
        self.croppedInputImageGX = self.inputImageGX[v1_y:v2_y + 1, v1_x: v2_x + 1]

        # Update origin coordinates, now in "real" global space
        x = max(0, x)
        y = max(0, y)

        # overlap_mask = self.output_img_filled_mask[y:y + self.croppedInput_h, x:x + self.croppedInput_w]
        #
        # no_overlap = (1 - overlap_mask).sum()

        no_overlap = self.count_non_overlap(x, y)

        if no_overlap == self.croppedInput_w * self.croppedInput_h:
            print("No overlap detected")
            self.direct_copy_input_to_output(y=y, x=x)
            return 0
        elif no_overlap == 0 and filling:
            print('Patch does not contribute in filling')
            return -1

        image = self.compute_cropped_output_image(y=y, x=x)

        imageGY, imageGX = compute_gradient_image(image)

        # Graph construction
        nb_nodes = self.croppedInput_w * self.croppedInput_h
        nb_edges = (self.croppedInput_w - 1) * (self.croppedInput_h - 1) * 2 + (self.croppedInput_w - 1) + (
                self.croppedInput_h - 1)
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
                            inputX=inputX,
                            inputY=inputY,
                            tX=tX,
                            tY=tY)

        flow = g.maxflow()

        self.process_seams(y, x, g)

        self.merge_pixels(y, x, g, imageGY, imageGX, blending)

        g = None
        self.seamNode = []

        return overlap

    def merge_pixels(self, y, x, g, imageGY, imageGX, blending):
        if blending:
            pass
        else:
            for j in range(self.croppedInput_h):
                for i in range(self.croppedInput_w):
                    node_nb_global = self.get_node_number_global(x, y, i, j)
                    the_node = self.globalNode[node_nb_global]
                    if self.globalNode[node_nb_global].empty:
                        # New pixel insertion
                        self.globalNode[node_nb_global].color = self.croppedInputImage[j, i]
                        self.globalNode[node_nb_global].empty = False
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
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                node_nb_global = self.get_node_number_global(x, y, i, j)
                nodeNbLocal = self.get_node_number_local(i, j)
                the_node = self.globalNode[node_nb_global]
                if i < self.croppedInput_w - 1:
                    if g.get_segment(nodeNbLocal) != g.get_segment(self.get_node_number_local(i + 1, j)):
                        the_node.newSeam = True

                if j < self.croppedInput_h - 1:
                    if g.get_segment(nodeNbLocal) != g.get_segment(self.get_node_number_local(i, j + 1)):
                        the_node.newSeam = True

                if len(self.seamNode) and (k < len(self.seamNode)) and (nodeNbLocal == self.seamNode[k].start):
                    # Process old seam
                    currentSeamNode = self.seamNode[k].seam
                    currentSeamNodeEnd = self.seamNode[k].end

                    if g.get_segment(nodeNbLocal) == source_type and g.get_segment(currentSeamNodeEnd) == sink_type:
                        # Old seam remains with new seam cost
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[node_nb_global].rightCost = self.seamNode[k].capacity3
                                self.globalNode[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[node_nb_global].bottomCost = self.seamNode[k].capacity3
                                self.globalNode[node_nb_global].seamBottom = True

                        else:
                            # Old seam remains with new seam cost
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[node_nb_global].rightCost = self.seamNode[k].capacity2
                                self.globalNode[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[node_nb_global].bottomCost = self.seamNode[k].capacity2
                                self.globalNode[node_nb_global].seamBottom = True
                    elif g.get_segment(nodeNbLocal) == sink_type and g.get_segment(currentSeamNodeEnd) == source_type:
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[node_nb_global].rightCost = self.seamNode[k].capacity2
                                self.globalNode[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[node_nb_global].bottomCost = self.seamNode[k].capacity2
                                self.globalNode[node_nb_global].seamBottom = True
                        else:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[node_nb_global].rightCost = self.seamNode[k].capacity3
                                self.globalNode[node_nb_global].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[node_nb_global].bottomCost = self.seamNode[k].capacity3
                                self.globalNode[node_nb_global].seamBottom = True
                    elif g.get_segment(currentSeamNode) == source_type:
                        if self.seamNode[k].orientation == 0:
                            self.globalNode[node_nb_global].rightCost = self.seamNode[k].capacity1
                            self.globalNode[node_nb_global].seamRight = True
                        else:
                            self.globalNode[node_nb_global].bottomCost = self.seamNode[k].capacity1
                            self.globalNode[node_nb_global].seamBottom = True

                    else:
                        pass

                    k += 1
                else:
                    # New seam
                    if i < self.croppedInput_w - 1:
                        if g.get_segment(nodeNbLocal) != g.get_segment(self.get_node_number_local(i + 1, j)):
                            self.globalNode[node_nb_global].seamRight = True

                    if j < self.croppedInput_h - 1:
                        if g.get_segment(nodeNbLocal) != g.get_segment(self.get_node_number_local(i, j + 1)):
                            self.globalNode[node_nb_global].seamBottom = True

    def add_seam_node(self, g, node_ids):
        for i in range(len(self.seamNode)):
            node_old_seam = node_ids[i]
            self.seamNode[i].seam = node_old_seam
            g.add_edge(self.seamNode[i].start, node_old_seam, self.seamNode[i].capacity2, self.seamNode[i].capacity2)
            g.add_edge(node_old_seam, self.seamNode[i].end, self.seamNode[i].capacity3, self.seamNode[i].capacity3)
            g.add_tedge(node_old_seam, 0.0, self.seamNode[i].capacity1)
            self.num_sink += 1

    def link_source_node(self, y, x, g):
        for i in range(self.croppedInput_w):
            j = 0
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.globalNode[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)

            j = self.croppedInput_h - 1
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.globalNode[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)

        for j in range(self.croppedInput_h):
            i = 0
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.globalNode[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)

            i = self.croppedInput_w - 1
            node_nb_global = self.get_node_number_global(x, y, i, j)
            if not self.globalNode[node_nb_global].empty:
                g.add_tedge(self.get_node_number_local(i, j), infiniteCap, 0.0)

    def link_sink_node(self, y, x, g, filling, random_refine, radius, inputX, inputY, tX, tY):
        if filling:
            print('Filling patch ', self.patch_number)
            for j in range(self.croppedInput_h):
                for i in range(self.croppedInput_w):
                    node_nb_global = self.get_node_number_global(x, y, i, j)
                    if not self.globalNode[node_nb_global].empty:
                        nodeNbGlobalLeft = self.get_node_number_global(x, y, i - 1, j)
                        nodeNbGlobalRight = self.get_node_number_global(x, y, i + 1, j)
                        nodeNbGlobalTop = self.get_node_number_global(x, y, i, j - 1)
                        nodeNbGlobalBottom = self.get_node_number_global(x, y, i, j + 1)

                        # empty left, right, top, bottom
                        if (nodeNbGlobalLeft != -1) and self.globalNode[nodeNbGlobalLeft].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalTop != -1) and self.globalNode[nodeNbGlobalTop].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalRight != -1) and self.globalNode[nodeNbGlobalRight].empty:
                            g.add_tedge(self.get_node_number_local(i, j), 0.0, infiniteCap)
                            self.num_sink += 1
                        elif (nodeNbGlobalBottom != -1) and self.globalNode[nodeNbGlobalBottom].empty:
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
                outputX = inputX + tX
                outputY = inputY + tY
                v1_x = max(0, outputX)
                v1_y = max(0, outputY)
                v2_x = min(self.output_width - 1, outputX + radius - 1)
                v2_y = min(self.output_height - 1, outputY + radius - 1)
                # Subpatch origin coordinates in output space
                outputX = v1_x
                outputY = v1_y
                # Subpatch origin coordinates in input space
                inputX = outputX - tX
                inputY = outputY - tY

                cj = 0
                ci = 0
                for j in range(v1_y, v2_y + 1):
                    for i in range(v1_x, v2_x + 1):
                        if (inputX + ci > 0) and (inputY + cj > 0):
                            node_nb_local = self.get_node_number_local(inputX + ci, inputY + cj)
                            g.add_tedge(node_nb_local, 0.0, infiniteCap)
                            self.num_sink += 1
                        ci += 1
                    cj += 1
            else:
                # random refinement
                if self.num_sink == 0:
                    # at least one pixel from new patch should be added
                    # so add the center pixel to sink
                    # this should not happen very often
                    j = self.croppedInput_h // 2
                    i = self.croppedInput_w // 2
                    node_nb_local = self.get_node_number_local(i, j)
                    # node_nb_global = self.getNodeNbGlobal(x, y, i, j)
                    g.add_tedge(node_nb_local, 0.0, infiniteCap)
                    self.num_sink += 1
                else:
                    print('already connected')

        assert self.num_sink > 0

    # -------------------------------- START: helper function --------------------------------

    def get_node_number_local(self, i: int, j: int) -> int:
        return i + self.croppedInput_w * j

    def get_node_number_global(self, x, y, i, j) -> int:
        if (x + i < self.output_width) and (x + i >= 0) and (y + j < self.output_height) and (y + j >= 0):
            return (x + i) + self.output_width * (y + j)
        else:
            return -1

    def add_img_border(self, img, border_y, border_x):
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_bordered = np.zeros((img_h + 2 * border_y, img_w + 2 * border_x, 3), dtype=int)
        img_bordered[border_y: border_y + img_h, border_x: border_x + img_w] = img
        return img_bordered

    def add_mask_border(self, mask, border_y, border_x):
        mask_h = mask.shape[0]
        mask_w = mask.shape[1]
        mask_bordered = np.zeros((mask_h + 2 * border_y, mask_w + 2 * border_x), dtype=int)
        mask_bordered[border_y: border_y + mask_h, border_x: border_x + mask_w] = mask
        return mask_bordered

    # -------------------------------- END: helper function --------------------------------

    def seamsErrorSubPatchRefinement(self, maxIter, radius, blending):

        inputRadius = radius
        maxErrX = 0
        maxErrY = 0
        x = 0
        y = 0
        err = 0.
        minErr = 0.
        minErrI = 0
        minErrJ = 0

        self.update_seams_max_error()

        for k in range(maxIter):
            if self.maxErrNodeNbGlobal != -1:
                radius = (inputRadius - 2) + (randint(0, 4))
                minErr = 1e15

                minErrI = -1
                minErrJ = -1

                maxErrX = self.maxErrNodeNbGlobal % self.output_width
                maxErrY = self.maxErrNodeNbGlobal // self.output_height

                # Fixed region position
                x = maxErrX - (radius // 2)
                y = maxErrY - (radius // 2)

                # Cropped fixed region
                v1X = max(0, x)
                v1Y = max(0, y)
                v2X = min(self.output_width - 1, x + radius - 1)
                v2Y = min(self.output_height - 1, y + radius - 1)
                cropped_w = v2X - v1X + 1
                cropped_h = v2Y - v1Y + 1

                for j in range(self.input_width - 3 * radius):
                    for i in self.input_height - 3 * radius:
                        err = self.computeLocalSSD(v1X - radius, v1Y - radius, i, j, 2 * radius + cropped_w,
                                                   2 * radius + cropped_h)
                        if (err < minErr):
                            minErr = err
                            minErrI = i
                            minErrJ = j

                minErrI += radius
                minErrJ += radius

    # -------------------------------- START: entire patch matching placement algorithm --------------------------------
    def entire_matching_filling(self, k, blending=False, save_img=False, show_seams=False):
        print('Initial synthesis: Entire Patch matching')
        overlap_width = self.input_width // 3
        overlap_height = self.input_height // 3

        offset_y = 0

        # y is sample in (offset_y-overlap_height, offset_y]
        y = offset_y - (overlap_height + randint(0, overlap_height - 1))

        while True:  # loop y
            print('New Row')
            offset_x = 0
            # sample x in [-overlap_width, 0)
            x = offset_x - (overlap_width + randint(0, overlap_width - 1))
            while True:  # loop x
                if y < self.output_height:
                    res = self.insert_patch(y, x)
                    if res != -1:  # some pixel is added
                        self.patch_number += 1
                        if save_img:
                            self.write_image()
                            if show_seams:
                                self.reveal_seams()
                            self.save_output_img(self.patch_number)

                offset_x += overlap_width  # go to next column

                # the x candidate is in [offset_x-overlap_width, offset_x)
                # the y candidate is in [offset_y-overlap_height, offset_y)

                # TODO: calculate cost matrix
                # TODO: sample y, x according to the cost matrix
                self.compute_entire_matching_cost_matrix()

                if x >= self.output_width:
                    break

            offset_y += overlap_height  # go to next row

            # sample y
            y = offset_y - (overlap_height + randint(0, overlap_height - 1))

            if y >= self.output_height:
                break

    # -------------------------------- END: entire patch matching placement algorithm --------------------------------

    # -------------------------------- START: sub patch matching placement algorithm --------------------------------

    # -------------------------------- END: sub patch matching placement algorithm --------------------------------

    # -------------------------------- START: random placement algorithm --------------------------------

    def random_fill(self, save_img=False, show_seams=False):
        print('Initial synthesis: Random')
        overlap_width = self.input_width // 3
        overlap_height = self.input_height // 3

        offset_y = 0

        # y is sample in (offset_y-overlap_height, offset_y]
        y = offset_y - (overlap_height + randint(0, overlap_height - 1))

        while True:  # loop y
            print('New Row')
            offset_x = 0
            # sample x in [-overlap_width, 0)
            x = offset_x - (overlap_width + randint(0, overlap_width - 1))
            while True:  # loop x
                if y < self.output_height:
                    res = self.insert_patch(y, x)
                    if res != -1:  # some pixel is added
                        self.patch_number += 1
                        if save_img:
                            self.write_image()
                            if show_seams:
                                self.reveal_seams()
                            self.save_output_img(self.patch_number)

                offset_x += overlap_width  # go to next column

                # sample x in [offset_x-overlap_width, offset_x)
                x = offset_x - (overlap_width + randint(0, overlap_width - 1))

                # sample y in [offset_y-overlap_height, offset_y)
                y = offset_y - (overlap_height + randint(0, overlap_height - 1))

                if x >= self.output_width:
                    break

            offset_y += overlap_height  # go to next row

            # sample y
            y = offset_y - (overlap_height + randint(0, overlap_height - 1))

            if y >= self.output_height:
                break

    def random_refinement(self, iter=20, save_img=False, show_seams=False):
        overlap_width = self.input_width // 3
        overlap_height = self.input_height // 3
        blending = False
        for k in range(iter):
            # sample y and x
            x = randint(-overlap_width, self.output_width - 1)
            y = randint(-overlap_height, self.output_height - 1)
            self.insert_patch(y, x, filling=False, blending=blending, random_refine=True)
            self.update_seams_max_error()
            self.patch_number += 1
            if save_img:
                self.write_image()
                if show_seams:
                    self.reveal_seams()
                self.save_output_img(self.patch_number)

    # -------------------------------- END: random placement algorithm --------------------------------

    # -------------------------------- START: image plot related functions --------------------------------
    def write_image(self):
        for j in range(self.output_height):
            for i in range(self.output_width):
                the_node = self.globalNode[self.get_node_number_global(0, 0, i, j)]
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
        name = os.path.join(out_dir, 'out_{}.png'.format(patch_id))
        plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.output_img)
        plt.savefig(name)

    # -------------------------------- END: image plot related functions --------------------------------

    def update_seams_max_error(self):
        nodeNbGlobal = 0
        self.maxErrNodeNbGlobal = -1
        maxErr = -1.0
        bs = self.borderSize
        for j in range(bs, self.output_height - bs):
            for i in range(bs, self.output_width - bs):
                nodeNbGlobal = self.get_node_number_global(0, 0, i, j)
                the_node = self.globalNode[nodeNbGlobal]
                if not the_node.empty:
                    if the_node.seamRight:
                        if the_node.rightCost > maxErr:
                            maxErr = the_node.rightCost
                            self.maxErrNodeNbGlobal = nodeNbGlobal

                    if the_node.seamBottom:
                        if the_node.bottomCost > maxErr:
                            maxErr = the_node.bottomCost
                            self.maxErrNodeNbGlobal = nodeNbGlobal

        return maxErr

    def computeSSD(self, x, y):
        v1X = max(0, x) - x
        v1Y = max(0, y) - y
        v2X = min(self.output_width - 1, x + self.input_width - 1) - x
        v2Y = min(self.output_height - 1, y + self.input_height - 1) - y

        self.croppedInput_w = v2X - v1X + 1
        self.croppedInput_h = v2Y - v1Y + 1
        x = max(0, x)
        y = max(0, y)
        err = 0.0
        nbpix = 0

        cj = 0
        for j in range(v1Y, v2Y):
            ci = 0
            for i in range(v1X, v2X):
                nodeNbGlobal = self.get_node_number_global(x, y, ci, cj)
                the_node = self.globalNode[nodeNbGlobal]
                if (not the_node.empty):
                    err += square_distance(the_node.color, self.input_img[j, i])
                    nbpix += 1
                ci += 1
            cj += 1
        if nbpix == 0:
            return 0
        else:
            return err / nbpix

    def compute_entire_matching_cost_matrix(self, y_start, y_end, x_start, x_end):
        matrix_height = y_end - y_start
        matrix_width = x_end - x_start


if __name__ == "__main__":
    img_in = imread('data/strawberries2.gif')
    # img_in = imread('data/green.gif')
    # img_in = imread('data/akeyboard_small.gif')
    if img_in.shape[2] == 4:
        # remove alpha channel
        img_in = np.array(img_in[:, :, 0:3])

    print('original image size: ', img_in.shape)

    out_dir = 'out/graph_cut_texture'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gc_texture = GraphCutTexture(img_in, img_in.shape[0] * 2, img_in.shape[1] * 2)

    placement = 'random'
    if placement not in ['random', 'entire', 'sub']:
        print('unknown placement method', )
        exit(1)

    if placement == 'random':
        gc_texture.random_fill(save_img=True, show_seams=True)
        gc_texture.random_refinement(iter=20, save_img=True, show_seams=True)
    elif placement == 'entire':
        gc_texture.entire_matching_filling(k=0.1)
        gc_texture.entire_matching_refinement(iter=20)
    elif placement == 'sub':
        gc_texture.sub_matching_filling(k=0.1)
        gc_texture.sub_matching_refinement(iter=20)

    gc_texture.write_image()
    # gc_texture.revealSeams()
    gc_texture.save_output_img('final')
    gc_texture.show_output_img()
