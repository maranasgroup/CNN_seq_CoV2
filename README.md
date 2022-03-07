# CNN_seq
  CNN model based on RBD-ACE2 sequence, predicting binding affinity change upon mutations.

## Configure environment
  Pre-trained models are based on these packages, and errors/warnings may occur when using different versions.

  Install Anaconda for Linux on the node following the [instructions](https://docs.anaconda.com/anaconda/install/linux/).
  ```bash
  conda env remove -n cnn_seq # Remove existing conda env
  conda create -n cnn_seq python=3.8.5 # Create new conda env
  conda activate cnn_seq # Activate the cnn_seq env
  conda install numpy matplotlib scikit-learn # Basic tools
  conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch # PyTorch with CUDA support
  conda install biopython -c conda-forge # Biopython
  ```

## Usage
  ### Prepare variants list for prediction.
  Modify or create the file "var_chk.csv"
  See "var_chk_circ.csv" for examples

  ### Predict K_{D,app} ratios for variants in "var_chk.csv"
  ```
  ./nn_kfold.py 5
  

  ```
