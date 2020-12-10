from imageio import imread
import numpy as np
import scipy
from matplotlib import pyplot as plt
from typing import Tuple
import maxflow
from random import random, randint
import scipy.ndimage as nd

plt.figure(num=None, figsize=(40, 32), dpi=80, facecolor='w', edgecolor='k')

minCap = 1e-7
infiniteCap = 1e12


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


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


class GraphCutTexture():
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
        # self.output_img_filled_mask = np.zeros((output_height, output_width), dtype=int)
        self.patch_number = 0

        self.globalNode = [GlobalNode() for i in range(output_width * output_height)]
        self.seamNode: [SeamNode] = []

        self.inputImageGY, self.inputImageGX = self.computeGradientImage(self.input_img)

    def insertPatch(self, y: int, x: int, filling=True):
        # y: new patch offset in global space, can be < 0
        sx = self.input_width
        sy = self.input_height

        nodeNbGlobal: int = 0

        v1X: int = max(0, x) - x
        v1Y: int = max(0, y) - y
        v2X: int = min(self.output_width - 1, x + sx - 1) - x
        v2Y: int = min(self.output_height - 1, y + sy - 1) - y
        self.croppedInput_w = v2X - v1X + 1
        self.croppedInput_h = v2Y - v1Y + 1

        print('Cropped patch size: {} * {}'.format(self.croppedInput_w, self.croppedInput_h))

        if self.croppedInput_w == 0 or self.croppedInput_h == 0:
            print("Patch lies outside output texture")
            return -1

        self.croppedInputImage = self.input_img[v1Y:v2Y + 1, v1X: v2X + 1]
        self.croppedInputImageGY = self.inputImageGY[v1Y:v2Y + 1, v1X: v2X + 1]
        self.croppedInputImageGX = self.inputImageGX[v1Y:v2Y + 1, v1X: v2X + 1]

        # Update origin coordinates, now in "real" global space
        x = max(0, x)
        y = max(0, y)

        # overlap_mask = self.output_img_filled_mask[y:y + self.croppedInput_h, x:x + self.croppedInput_w]
        #
        # noOverlap = (1 - overlap_mask).sum()

        noOverlap = 0
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                if (self.globalNode[nodeNbGlobal].empty):
                    noOverlap += 1

        if noOverlap == self.croppedInput_w * self.croppedInput_h:
            print("No overlap detected")
            # self.copy_to_offset(self.output_img, self.croppedInputImage, (y,x))
            # self.copy_to_offset(self.output_img_filled_mask, self.croppedInputImage, (y,x))

            for j in range(self.croppedInput_h):
                for i in range(self.croppedInput_w):
                    nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                    theNode = self.globalNode[nodeNbGlobal]
                    if theNode.empty:
                        theNode.color = self.croppedInputImage[j, i]
                        theNode.empty = False
            return 0
        elif (noOverlap == 0):
            print('Patch does not contribute in filling')
            return -1

        image = np.zeros_like(self.croppedInputImage)

        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                if not self.globalNode[nodeNbGlobal].empty:
                    image[j, i] = self.globalNode[nodeNbGlobal].color
                else:
                    image[j, i] = [0, 0, 0]

        imageGY, imageGX = self.computeGradientImage(image)

        # Graph construction
        nbNodes = self.croppedInput_w * self.croppedInput_h
        nbEdges = (self.croppedInput_w - 1) * (self.croppedInput_h - 1) * 2 + (self.croppedInput_w - 1) + (
                self.croppedInput_h - 1)
        nbSeamNodes = nbEdges
        nbEdges += nbSeamNodes

        g = maxflow.Graph[float](nbNodes + 2 * nbSeamNodes, nbEdges)

        g.add_nodes(nbNodes)

        nodeNbGlobalRight: int = 0
        nodeNbGlobalBottom: int = 0

        nodeNbLocal: int = 0
        nodeNbLocalRight: int = 0
        nodeNbLocalBottom: int = 0

        grad = 0.
        d1 = 0.
        d2 = 0.
        d3 = 0.
        d4 = 0.
        capRight = 0.
        capBottom = 0.
        double = 0.
        capacity1 = 0.
        capacity2 = 0.
        capacity3 = 0.

        overlap: int = 0
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                nodeNbLocal = self.getNodeNbLocal(i, j)

                if (i < self.croppedInput_w - 1):
                    nodeNbGlobalRight = self.getNodeNbGlobal(x, y, i + 1, j)
                    nodeNbLocalRight = self.getNodeNbLocal(i + 1, j)

                    if not self.globalNode[nodeNbGlobal].empty:
                        if not self.globalNode[nodeNbGlobalRight].empty:  # right
                            d1 = self.distColor(self.globalNode[nodeNbGlobal].color, self.croppedInputImage[j, i])
                            d2 = self.distColor(self.globalNode[nodeNbGlobalRight].color,
                                                self.croppedInputImage[j, i + 1])
                            if self.globalNode[nodeNbGlobal].seamRight:
                                # Old seam: a seam node will created
                                capacity1 = self.globalNode[nodeNbGlobal].rightCost
                                d3 = self.distColor(self.globalNode[nodeNbGlobalRight].colorOtherPatch,
                                                    self.croppedInputImage[j, i + 1])
                                d4 = self.distColor(self.croppedInputImage[j, i],
                                                    self.globalNode[nodeNbGlobal].colorOtherPatch)
                                grad = ((self.croppedInputImageGX[j, i] / 255.0)
                                        + (self.croppedInputImageGX[j, i + 1] / 255.0)
                                        + (imageGX[j, i] / 255.0)
                                        + (self.globalNode[nodeNbGlobalRight].gradXOtherPatch / 255.0))
                                grad += 1.0

                                capacity2 = (d1 + d3) / grad
                                grad = (self.croppedInputImageGX[j, i] / 255.0) \
                                       + (self.croppedInputImageGX[j, i + 1] / 255.0) \
                                       + (imageGX[j, i + 1] / 255.0) \
                                       + (self.globalNode[nodeNbGlobal].gradXOtherPatch / 255.0)
                                grad += 1.0

                                capacity3 = (d4 + d2) / grad
                                capacity2 += minCap
                                capacity3 += minCap
                                self.seamNode.append(
                                    SeamNode(nodeNbLocal, nodeNbLocalRight, capacity1, capacity2, capacity3, 0))
                            else:
                                # No old seam
                                grad = (self.croppedInputImageGX[j, i] / 255.0) \
                                       + (self.croppedInputImageGX[j, i + 1] / 255.0) \
                                       + (imageGX[j, i] / 255.0) \
                                       + (imageGX[j, i + 1] / 255.0)
                                grad += 1.0
                                capRight = (d1 + d2) / grad
                                capRight += minCap
                                g.add_edge(nodeNbLocal, nodeNbLocalRight, capRight, capRight)
                                self.globalNode[nodeNbGlobal].rightCost = capRight
                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(nodeNbLocal, nodeNbLocalRight, 0.0, 0.0)
                            self.globalNode[nodeNbGlobal].rightCost = 0.0
                    else:
                        g.add_edge(nodeNbLocal, nodeNbLocalRight, 0.0, 0.0)
                        self.globalNode[nodeNbGlobal].rightCost = 0.0

                if (j < self.croppedInput_h - 1):
                    nodeNbGlobalBottom = self.getNodeNbGlobal(x, y, i, j + 1)
                    nodeNbLocalBottom = self.getNodeNbLocal(i, j + 1)

                    if not self.globalNode[nodeNbGlobal].empty:
                        if not self.globalNode[nodeNbGlobalBottom].empty:  # bottom
                            # Overlap
                            d1 = self.distColor(self.globalNode[nodeNbGlobal].color, self.croppedInputImage[j, i])
                            d2 = self.distColor(self.globalNode[nodeNbGlobalBottom].color,
                                                self.croppedInputImage[j + 1, i])
                            if self.globalNode[nodeNbGlobal].seamBottom:
                                # Old seam: a seam node will created
                                capacity1 = self.globalNode[nodeNbGlobal].bottomCost

                                d3 = self.distColor(self.globalNode[nodeNbGlobalBottom].colorOtherPatch,
                                                    self.croppedInputImage[j + 1, i])
                                d4 = self.distColor(self.croppedInputImage[j, i],
                                                    self.globalNode[nodeNbGlobal].colorOtherPatch)
                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (imageGY[j, i] / 255.0) \
                                       + (self.globalNode[nodeNbGlobalBottom].gradYOtherPatch / 255.0)
                                grad += 1.0
                                capacity2 = (d1 + d3) / grad

                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (imageGY[j + 1, i] / 255.0) \
                                       + (self.globalNode[nodeNbGlobal].gradYOtherPatch / 255.0)
                                grad += 1.0
                                capacity3 = (d4 + d2) / grad

                                capacity2 += minCap
                                capacity3 += minCap
                                self.seamNode.append(
                                    SeamNode(nodeNbLocal, nodeNbLocalBottom, capacity1, capacity2, capacity3, 1))
                            else:
                                # No old seam
                                grad = (self.croppedInputImageGY[j, i] / 255.0) \
                                       + (self.croppedInputImageGY[j + 1, i] / 255.0) \
                                       + (imageGX[j, i] / 255.0) \
                                       + (imageGX[j + 1, i] / 255.0)
                                grad += 1.0
                                capBottom = (d1 + d2) / grad
                                capBottom += minCap
                                g.add_edge(nodeNbLocal, nodeNbLocalBottom, capBottom, capBottom)
                                self.globalNode[nodeNbGlobal].bottomCost = capBottom

                            overlap += 1

                        else:
                            # No overlap
                            g.add_edge(nodeNbLocal, nodeNbLocalBottom, 0.0, 0.0)
                            self.globalNode[nodeNbGlobal].bottomCost = 0.0
                    else:
                        # No overlap
                        g.add_edge(nodeNbLocal, nodeNbLocalBottom, 0.0, 0.0)
                        self.globalNode[nodeNbGlobal].bottomCost = 0.0

        print("Number of seam nodes: {}".format(len(self.seamNode)))
        nodeOldSeams = g.add_nodes(len(self.seamNode))

        for i in range(len(self.seamNode)):
            nodeOldSeam = nodeOldSeams[i]
            self.seamNode[i].seam = nodeOldSeam
            g.add_edge(self.seamNode[i].start, nodeOldSeam, self.seamNode[i].capacity2, self.seamNode[i].capacity2)
            g.add_edge(nodeOldSeam, self.seamNode[i].end, self.seamNode[i].capacity3, self.seamNode[i].capacity3)
            g.add_tedge(nodeOldSeam, 0.0, self.seamNode[i].capacity1)

        print('Graph created')

        # Assignments to source node

        for i in range(self.croppedInput_w):
            j = 0
            nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
            if not self.globalNode[nodeNbGlobal].empty:
                g.add_tedge(self.getNodeNbLocal(i, j), infiniteCap, 0.0)

            j = self.croppedInput_h - 1
            nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
            if not self.globalNode[nodeNbGlobal].empty:
                g.add_tedge(self.getNodeNbLocal(i, j), infiniteCap, 0.0)

        for j in range(self.croppedInput_h):
            i = 0
            nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
            if not self.globalNode[nodeNbGlobal].empty:
                g.add_tedge(self.getNodeNbLocal(i, j), infiniteCap, 0.0)

            i = self.croppedInput_w - 1
            nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
            if not self.globalNode[nodeNbGlobal].empty:
                g.add_tedge(self.getNodeNbLocal(i, j), infiniteCap, 0.0)

        nbSink = 0
        if filling:
            for j in range(self.croppedInput_h):
                for i in range(self.croppedInput_w):
                    nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                    if not self.globalNode[nodeNbGlobal].empty:
                        nodeNbGlobalLeft = self.getNodeNbGlobal(x, y, i - 1, j)
                        nodeNbGlobalRight = self.getNodeNbGlobal(x, y, i + 1, j)
                        nodeNbGlobalTop = self.getNodeNbGlobal(x, y, i, j - 1)
                        nodeNbGlobalBottom = self.getNodeNbGlobal(x, y, i, j + 1)

                        if ((nodeNbGlobalLeft != -1) and self.globalNode[nodeNbGlobalLeft].empty):
                            g.add_tedge(self.getNodeNbLocal(i, j), 0.0, infiniteCap)
                            nbSink += 1
                        elif ((nodeNbGlobalTop != -1) and self.globalNode[nodeNbGlobalTop].empty):
                            g.add_tedge(self.getNodeNbLocal(i, j), 0.0, infiniteCap)
                            nbSink += 1
                        elif ((nodeNbGlobalRight != -1) and self.globalNode[nodeNbGlobalRight].empty):
                            g.add_tedge(self.getNodeNbLocal(i, j), 0.0, infiniteCap)
                            nbSink += 1
                        elif ((nodeNbGlobalBottom != -1) and self.globalNode[nodeNbGlobalBottom].empty):
                            g.add_tedge(self.getNodeNbLocal(i, j), 0.0, infiniteCap)
                            nbSink += 1

        else:
            # Refinement
            print('NOT IMPLEMENTED: Refinement')

        assert (nbSink)
        flow = g.maxflow()
        print('maxflow', flow)

        currentSeamNode = 0
        currentSeamNodeEnd = 0

        k = 0

        source_type = 0
        sink_type = 1
        for j in range(self.croppedInput_h):
            for i in range(self.croppedInput_w):
                nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                nodeNbLocal = self.getNodeNbLocal(i, j)

                if i < self.croppedInput_w - 1:
                    if g.get_segment(nodeNbLocal) != g.get_segment(self.getNodeNbLocal(i + 1, j)):
                        self.globalNode[nodeNbGlobal].newSeam = True

                if j < self.croppedInput_h - 1:
                    if g.get_segment(nodeNbLocal) != g.get_segment(self.getNodeNbLocal(i, j + 1)):
                        self.globalNode[nodeNbGlobal].newSeam = True

                if len(self.seamNode) and (k < len(self.seamNode)) and (nodeNbLocal == self.seamNode[k].start):
                    # Process old seam

                    currentSeamNode = self.seamNode[k].seam
                    currentSeamNodeEnd = self.seamNode[k].end

                    if g.get_segment(nodeNbLocal) == source_type and g.get_segment(currentSeamNodeEnd) == sink_type:
                        # Old seam remains with new seam cost
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[nodeNbGlobal].rightCost = self.seamNode[k].capacity3
                                self.globalNode[nodeNbGlobal].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[nodeNbGlobal].bottomCost = self.seamNode[k].capacity3
                                self.globalNode[nodeNbGlobal].seamBottom = True

                        else:
                            # Old seam remains with new seam cost
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[nodeNbGlobal].rightCost = self.seamNode[k].capacity2
                                self.globalNode[nodeNbGlobal].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[nodeNbGlobal].bottomCost = self.seamNode[k].capacity2
                                self.globalNode[nodeNbGlobal].seamBottom = True
                    elif g.get_segment(nodeNbLocal) == sink_type and g.get_segment(currentSeamNodeEnd) == source_type:
                        if g.get_segment(currentSeamNode) == source_type:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[nodeNbGlobal].rightCost = self.seamNode[k].capacity2
                                self.globalNode[nodeNbGlobal].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[nodeNbGlobal].bottomCost = self.seamNode[k].capacity2
                                self.globalNode[nodeNbGlobal].seamBottom = True
                        else:
                            if self.seamNode[k].orientation == 0:
                                # Right
                                self.globalNode[nodeNbGlobal].rightCost = self.seamNode[k].capacity3
                                self.globalNode[nodeNbGlobal].seamRight = True
                            else:
                                # Bottom
                                self.globalNode[nodeNbGlobal].bottomCost = self.seamNode[k].capacity3
                                self.globalNode[nodeNbGlobal].seamBottom = True
                    elif g.get_segment(currentSeamNode) == source_type:
                        if self.seamNode[k].orientation == 0:
                            self.globalNode[nodeNbGlobal].rightCost = self.seamNode[k].capacity1
                            self.globalNode[nodeNbGlobal].seamRight = True
                        else:
                            self.globalNode[nodeNbGlobal].bottomCost = self.seamNode[k].capacity1
                            self.globalNode[nodeNbGlobal].seamBottom = True

                    else:
                        pass

                    k += 1
                else:
                    # New seam
                    if i < self.croppedInput_w - 1:
                        if g.get_segment(nodeNbLocal) != g.get_segment(self.getNodeNbLocal(i + 1, j)):
                            self.globalNode[nodeNbGlobal].seamRight = True

                    if j < self.croppedInput_h - 1:
                        if g.get_segment(nodeNbLocal) != g.get_segment(self.getNodeNbLocal(i, j + 1)):
                            self.globalNode[nodeNbGlobal].seamBottom = True

        blending = False

        if blending:
            pass
        else:
            for j in range(self.croppedInput_h):
                for i in range(self.croppedInput_w):
                    nodeNbGlobal = self.getNodeNbGlobal(x, y, i, j)
                    theNode = self.globalNode[nodeNbGlobal]
                    if self.globalNode[nodeNbGlobal].empty:
                        # New pixel insertion
                        self.globalNode[nodeNbGlobal].color = self.croppedInputImage[j, i]
                        self.globalNode[nodeNbGlobal].empty = False
                    else:
                        if g.get_segment(self.getNodeNbLocal(i, j)) == source_type:
                            theNode.colorOtherPatch = self.croppedInputImage[j, i]
                            theNode.gradXOtherPatch = self.croppedInputImageGX[j, i]
                            theNode.gradYOtherPatch = self.croppedInputImageGY[j, i]

                        else:
                            theNode.colorOtherPatch = theNode.color
                            theNode.gradXOtherPatch = imageGX[j, i]
                            theNode.gradYOtherPatch = imageGY[j, i]
                            theNode.color = self.croppedInputImage[j, i]
                            theNode.empty = False

                            if not theNode.newSeam:
                                if theNode.seamRight:
                                    theNode.seamRight = False
                                if theNode.seamBottom:
                                    theNode.seamBottom = False

                    theNode.newSeam = False

        g = None
        self.seamNode = []

        return overlap

    def computeGradientImage(self, image):
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

    def distColor(self, c1, c2):
        return (np.abs(c1 - c2)).sum()

    def getNodeNbLocal(self, i: int, j: int) -> int:
        return i + self.croppedInput_w * j

    def getNodeNbGlobal(self, x, y, i, j) -> int:
        if (x + i < self.output_width) and (x + i >= 0) and (y + j < self.output_height) and (y + j >= 0):
            return (x + i) + self.output_width * (y + j)
        else:
            return -1

    def get_nodeids_connected_to_old_and_new(self, nodeids, old_mask):
        # neighbour_kernel = np.array([[0, 1, 0],
        #                              [1, 0, 1],
        #                              [0, 1, 0]])  # up, down, left, right
        # neighbour_average = nd.convolve(old_mask, neighbour_kernel, mode='constant')
        # on_edge = (neighbour_average != old_mask)

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
        return match_cost_right

    def construct_cost_matrix_down(self, overlap_new: np.array, overlap_old: np.array):
        difference_between_patch = np.abs(overlap_new - overlap_old)
        shift_up_dif = np.roll(difference_between_patch, (-1, 0))
        match_cost_down = (difference_between_patch + shift_up_dif).sum(axis=2)
        return match_cost_down

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
                    res = self.insertPatch(y, x)
                    if res != -10:
                        self.patch_number += 1
                        self.writeImage()
                        self.save_output_img(self.patch_number)
                        # self.show_output_img()

                x = x + (overlap_width + randint(0, overlap_width - 1))
                y = offset_y - (overlap_height + randint(0, overlap_height - 1))

                if x >= self.output_width:
                    break

            offset_y += overlap_height
            y = offset_y - (overlap_height + randint(0, overlap_height - 1))

            if y >= self.output_height:
                break

    def writeImage(self):
        for j in range(self.output_height):
            for i in range(self.output_width):
                theNode = self.globalNode[self.getNodeNbGlobal(0, 0, i, j)]
                if not theNode.empty:
                    self.output_img[j, i] = theNode.color
                else:
                    self.output_img[j, i] = [0, 0, 0]

    def show_output_img(self):
        plt.subplot(1, 1, 1)
        plt.imshow(self.output_img)
        plt.show()

    def set_img(self, img, index, title=None):
        # ax = plt.subplot(3, 2, index)
        # ax.set_title(title)
        # plt.imshow(img)
        pass

    def save_output_img(self, patch_id):
        name = 'out_{}.png'.format(patch_id)
        plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.output_img)
        plt.savefig(name)


if __name__ == "__main__":
    # img_in = imread('data/strawberries2.gif')
    # img_in = imread('data/green.gif')
    img_in = imread('data/akeyboard_small.gif')
    if img_in.shape[2] == 4:
        # remove alpha channel
        img_in = np.array(img_in[:, :, 0:3])
    # plt.imshow(img_in)
    # plt.show()

    print('original image size: ', img_in.shape)

    gc_texture = GraphCutTexture(img_in, img_in.shape[0] * 2, img_in.shape[1] * 2)

    gc_texture.random_fill()

    gc_texture.writeImage()
    gc_texture.save_output_img('final')
    gc_texture.show_output_img()
