# Object_detection
Descriptors + PCA + classifier + salience analysis + sliding window etc.  

This branch (wip/svm_hog_pca) enhances the project introducing Principal Component Analysis (PCA)
for feature space dimensionality reduction and denoising.

Usage with PCA:  
Training
python hog_svm_pca.py  -p ./../learn_data/wheel/objects -n ./../learn_data/wheel/backs
-m ./../dat/wheels.dat -o ./../dat/pca.src

Detection
python slide_window_pca.py -m ./../dat/wheels.dat -i ./test_data/T00001_00001_1_2height_cubic.png 
-o ./../dat/pca.src -c svm
