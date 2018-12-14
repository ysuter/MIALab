"""The feature extraction module contains classes for feature extraction."""
import sys

import numpy as np
import pymia.filtering.filter as fltr
import SimpleITK as sitk
import cv2
import math


class AtlasCoordinates(fltr.IFilter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.IFilterParams = None) -> sitk.Image:
        """Executes a atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # create matrix with homogenous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # generate transformation matrix
        tmpmat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmpmat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'AtlasCoordinates:\n' \
            .format(self=self)


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features:

            - mean
            - variance
            - sigma
            - skewness
            - kurtosis
            - entropy
            - energy
            - snr
            - min
            - max
            - range
            - percentile10th
            - percentile25th
            - percentile50th
            - percentile75th
            - percentile90th
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    numvalues = len(values)
    p = values / (np.sum(values) + eps)
    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(numvalues * (numvalues - 1)) / (numvalues - 2) * np.sum((values - mean) ** 3) /
                     (numvalues*std**3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (numvalues * std ** 4 + eps),  # kurtosis
                     np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p**2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90)
                     ])

class HOGFeatureFilter():

    def __init__(self, img: sitk.Image, cell_size=16, bin_size=8):
    #def __init__(self, cell_size=16, bin_size=8):

        self.img = sitk.GetArrayFromImage(img)
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size  # Agila, you needed to change this line by replacing '/' to '//' (returns and integer instead of float)
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self, img):
        height, width= img.shape
        gradient_magnitude, gradient_angle = self.global_gradient(img)
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))  # Agila, you needed to change this line by replacing '/' to '//' (returns and integer instead of float)
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):

                 cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                  j * self.cell_size:(j + 1) * self.cell_size]
                 cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                              j * self.cell_size:(j + 1) * self.cell_size]
                 cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                   normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                   block_vector = normalize(block_vector, magnitude)
                   hog_vector.append(block_vector)
        return hog_vector, hog_image


    def execute(self, image: sitk.Image, params: fltr.IFilterParams = None) -> sitk.Image:

        img_arr = sitk.GetArrayFromImage(image)

        depth, height, width= img_arr.shape
        img_arr = np.sqrt(img_arr / float(np.max(img_arr)))
        img_arr *= 255

        hog_slicestack = np.empty((depth, height, width))

        # slice-wise extraction
        for idx in range(1, depth):
            print('Slice: ' + str(idx))
            img_arr_slice = img_arr[idx, :, :]
            hog_vector, hog_slicestack[idx, :, :] = self.extract(img_arr_slice)

        hog_featimg = sitk.GetImageFromArray(hog_slicestack)
        hog_featimg.CopyInformation(image)
        return hog_featimg

    def global_gradient(self, img):
        gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


class NeighborhoodFeatureExtractor(fltr.IFilter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.neighborhood_radius = 3
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.IFilterParams=None) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if np.isscalar(function_output):
            img_out = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        elif not isinstance(function_output, np.ndarray):
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.ndim > 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.shape[0] <= 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray with at least two elements')
        else:
            img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])

        img_out_arr = sitk.GetArrayFromImage(img_out)
        img_arr = sitk.GetArrayFromImage(image)
        z, y, x = img_arr.shape

        z_offset = self.kernel[2]
        y_offset = self.kernel[1]
        x_offset = self.kernel[0]
        pad = ((0, z_offset), (0, y_offset), (0, x_offset))
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')

        for xx in range(x):
            for yy in range(y):
                for zz in range(z):

                    val = self.function(img_arr_padded[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
                    img_out_arr[zz, yy, xx] = val

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'NeighborhoodFeatureExtractor:\n' \
            .format(self=self)


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator.

    A training mask is an image with intensity values 0 and 1, where 1 represents masked.
    Such a mask can be used to sample voxels for training.
    """

    @staticmethod
    def get_mask(ground_truth: sitk.Image,
                 ground_truth_labels: list,
                 label_percentages: list,
                 background_mask: sitk.Image=None) -> sitk.Image:
        """Gets a training mask.

        Args:
            ground_truth (sitk.Image): The ground truth image.
            ground_truth_labels (list of int): The ground truth labels,
                where 0=background, 1=label1, 2=label2, ..., e.g. [0, 1]
            label_percentages (list of float): The percentage of voxels of a corresponding label to extract as mask,
                e.g. [0.2, 0.2].
            background_mask (sitk.Image): A mask, where intensity 0 indicates voxels to exclude independent of the label.

        Returns:
            sitk.Image: The training mask.
        """

        # initialize mask
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)  # convert to float because of np.nan
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x = indices[no][0]
                y = indices[no][1]
                z = indices[no][2]
                mask_array[x, y, z] = 1  # this is a masked item

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetOrigin(ground_truth.GetOrigin())
        mask.SetDirection(ground_truth.GetDirection())
        mask.SetSpacing(ground_truth.GetSpacing())

        return mask
