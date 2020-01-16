from preprocessing_imgs import load_data, PreprocessingUnit
import cv2

data, masks = load_data("data/test/")

prep = PreprocessingUnit(data, masks)
blurred = prep.blurring()
binarized = prep.binarization(blurred)
v = prep.segmentation(binarized)
print(v)




cv2.destroyAllWindows()