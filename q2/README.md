Project 1: Conditional Random Fields for Structured Output Prediction

This project aims towards implementing a conditional random field for optical character recognition (OCR) with emphasis on inference and performance test. The project is divided into four parts:

1. Conditional Random Fields: We derive the gradients with respect to weights Wj and transition matrix Tij. We then show the gradient of log(Z) is exactly the expectation of features. Finally, we implement a decoder with computational cost O(m(y)^2) using the max-sum algorithm. We also implemented a brute-force solution in time O((y)^m) to test that decoder implementation is correct.
2. Training Conditional Random Fields: We now implement a DP algorithm to compute log(p(y|X)) and its gradient with respect to Wj and Tij. We used scipy.optimize.check_grad to ensure the correctness of the computed gradient. The second part of this portion dealt with using fmin_bfgs to get the optimal values for Wj and Tij. Finally, we used the decoder from part 1 and used the new weights to predict labels for characters in test.txt.
3. Benchmarking with Other Methods: We benchmarked our implementation with other methods such as SVM-Struct and SVM-MC and plotted the letter-wise and word-wise prediction accuracy on test data.
4. Robustness to Tampering: We explore what effects do rotation and translation of images have on the decoder and its accuracy. We ran the CRF and SVM-MC algorithm on the distorted images and checked the letter-wise and word-wise prediction accuracy on test data.

Prerequisities

Packages required to run the code can be found in requirements.txt:

Run: pip install -r requirements.txt

1. scipy
2. scikit-learn
3. numpy

Run each script for respective question.
Comment out parts which are long running.

Note: TF is for checking with our final solution, to run, use 
pip install tensorflow or 
pip install tensorflow-gpu if CUDA is installed

Running the tests

1. Run q1.py to run the max-sum decoder on 100 characters from the initial dataset. The results are stored in result/decoder_output.txt.
2. Run q2.py to calculate log(p(y|X)), calculate the gradients, and start the optimizer. The gradients can be found in result/gradient.txt. The predictions from 2(b) can be found in result/prediction.txt
3. The optimal parameters are pre-computed based on C values and can be found under result/solution[C].txt where C=1, 10, 100, 1000.
4. The optimal paramters are pre-computed and stored under result/solution[C]distortion.txt where C=500, 1000, 1500, 2000.

Note: Files q1_tf.py, q2_tf.py, q3_tf.py are Tensorflow implementations that we used to cross-check and verify our manual implementations. You can run those files to check the various outcomes.

Acknowledgements

Professor Xinhua Zhang
Vignesh Ganapathiraman
