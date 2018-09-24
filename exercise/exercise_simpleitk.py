import SimpleITK as sitk

import exercise.helper as helper


def main():
    callback = helper.TestCallback()
    callback.start('SimpleITK')

    callback.start_test('load_image')
    img = load_image('../data/exercise/subjectX/T1native.nii.gz', False)
    load_ok = img.GetPixelID() == 8 and img.GetSize() == (181, 217, 181) and img.GetPixel(100, 100, 100) == 12175 and \
              img.GetPixel(100, 100, 101) == 11972
    callback.end_test(load_ok)

    callback.start_test('to_numpy_array')
    np_img = to_numpy_array(img)
    to_numpy_ok = np_img.dtype.name == 'float32' and np_img.shape == (181, 217, 181) and np_img[100, 100, 100] == 12175 \
                  and np_img[101, 100, 100] == 11972
    callback.end_test(to_numpy_ok)

    callback.start_test('to_sitk_image')
    rev_img = to_sitk_image(np_img, img)
    to_sitk_ok = rev_img.GetOrigin() == img.GetOrigin() and rev_img.GetSpacing() == img.GetSpacing() and \
                 rev_img.GetDirection() == img.GetDirection() and rev_img.GetPixel(100, 100, 100) == 12175 and \
                 rev_img.GetPixel(100, 100, 101) == 11972
    callback.end_test(to_sitk_ok)

    callback.start_test('register_images')
    atlas_img = load_image('../data/exercise/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz', False)
    label_img = load_image('../data/exercise/subjectX/labels_native.nii.gz', True)
    registered_img, registered_label = register_images(img, label_img, atlas_img)
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(registered_img, registered_label)
    labels = stats.GetLabels()
    register_ok = registered_img.GetSize() == registered_label.GetSize() == (197, 233, 189) and labels == tuple(range(6))
    callback.end_test(register_ok)

    callback.start_test('preprocss_rescale_numpy')
    pre_np = preprocess_rescale_numpy(np_img, -3, 101)
    pre_np_ok = pre_np.min() == -3 and pre_np.max() == 101
    callback.end_test(pre_np_ok)

    callback.start_test('preprocss_rescale_sitk')
    pre_sitk = preprocess_rescale_sitk(img, -3, 101)
    min_max = sitk.MinimumMaximumImageFilter()
    min_max.Execute(pre_sitk)
    pre_sitk_ok = min_max.GetMinimum() == -3 and min_max.GetMaximum() == 101
    callback.end_test(pre_sitk_ok)

    callback.start_test('extract_feature_median')
    median_img = extract_feature_median(img)
    median_ref = load_image('../data/exercise/subjectX/T1med.nii.gz', False)
    min_max = sitk.MinimumMaximumImageFilter()
    min_max.Execute(median_img - median_ref)
    median_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
    callback.end_test(median_ok)

    callback.start_test('postprocess_largest_component')
    largest_hippocampus = postprocess_largest_component(label_img == 3)  # 3: hippocampus
    largest_ref = load_image('../data/exercise/subjectX/hippocampus_largest.nii.gz', True)
    min_max = sitk.MinimumMaximumImageFilter()
    min_max.Execute(largest_hippocampus - largest_ref)
    post_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
    callback.end_test(post_ok)

    callback.end()


def _get_registration_method(atlas_img, img) -> sitk.ImageRegistrationMethod:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.2)

    registration_method.SetMetricUseFixedImageGradientFilter(False)
    registration_method.SetMetricUseMovingImageGradientFilter(False)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(atlas_img, img,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    return registration_method


def load_image(img_path, is_label_img):
    # todo: load the image from the image path with the SimpleITK interface (hint: 'ReadImage')
    # todo: if 'is_label_img' is True use outputPixelType=sitk.sitkUInt8, else use outputPixelType=sitk.sitkFloat32

    pixel_type = None  # todo: modify here
    img = None  # todo: modify here

    return img


def to_numpy_array(img):
    # todo: transform the SimpleITK image to a numpy ndarray (hint: 'GetArrayFromImage')
    np_img = None  # todo: modify here
    return np_img


def to_sitk_image(np_image, orig_img):
    # todo: transform the numpy ndarray to a SimpleITK image (hint: 'GetImageFromArray')
    # todo: do not forget to copy meta-information (e.g. spacing, origin, etc.) from the reference image (hint: 'CopyInformation')!!!
    #       (otherwise defaults are set)

    img = None  # todo: modify here
    # todo: ...

    return img


def register_images(img, label_img, atlas_img):
    registration_method = _get_registration_method(atlas_img, img)
    # todo: apply the registration_method to the img (hint: fixed=atlas_img, moving=img)
    transform = None  # todo: modify here

    # todo: apply the obtained transform (img to atlas_img)
    # hint: 'Resample' (with referenceImage=atlas_img, transform=transform, defaultPixelValue=0.0, interpolator=sitkLinear, outputPixelType=label_img.GetPixelIDValue())
    registered_img = None  # todo: modify here

    # todo: apply the obtained transform to register the label image (label_img) to the atlas, too
    # hint: 'Resample' (with defaultPixelValue=0.0, interpolator=sitkNearestNeighbor, outputPixelType=label_img.GetPixelIDValue())
    registered_label = None  # todo: modify here

    return registered_img, registered_label


def preprocess_rescale_numpy(np_img, new_min_val, new_max_val):
    max_val = np_img.max()
    min_val = np_img.min()
    # todo: rescale the intensities of the np_img to the range [new_min_val, new_max_val]. Use numpy arithmetics only.
    rescaled_np_img = None  # todo: modify here

    return rescaled_np_img


def preprocess_rescale_sitk(img, new_min_val, new_max_val):
    # todo: rescale the intensities of the img to the range [new_min_val, new_max_val] (hint: RescaleIntensity)
    rescaled_img = None  # todo: modify here

    return rescaled_img


def extract_feature_median(img):
    # todo: apply median filter to image (hint: 'Median')
    median_img = None  # todo: modify here

    return median_img


def postprocess_largest_component(label_img):
    # todo: get the connected components from the label_img
    connected_components = None  # todo: modify here

    # todo: order the component by ascending component size (hint: 'RelabelComponent')
    relabeled_components = None  # todo: modify here

    largest_component = relabeled_components == 1  # zero is background
    return largest_component


if __name__ == '__main__':
    main()