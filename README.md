# CNN_seq
  CNN model based on RBD-ACE2 sequence, predicting binding affinity change upon mutations.

# Configure environment
  Pre-trained models are based on these version, error may occur for different versions of packages

  ```bash
    conda env remove -n cnn_seq
    conda create -n cnn_seq python=3.8.5
    conda activate cnn_seq
    conda install numpy matplotlib scikit-learn
    conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install biopython
  ```
