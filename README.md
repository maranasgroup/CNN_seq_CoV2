# CNN_seq
  CNN model based on RBD-ACE2 sequence, predicting binding affinity change upon mutations.

## Configure environment
  The code currently only supports CUDA-supported machine.

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

## Files
  [./]
  `cnn_kfold.py` is the main code for training, prediction and analysls.
  `var_blind_test_set.csv` contains 1677 variants, constituting a completely blind test set.
  `var_chk.csv` contains all variants to be predicted.
  `var_chk_circ.csv` contains many variants (including circulating variants), which can serve as an example of how to prepare the list.

  [./Mater]
  [./Mater/expt_multi_mutant] Contains experimental data for training purposes.
  [./Mater/python] Contains python modules called by `cnn_kfold.py`.
  [./Mater/species] Contains basic information used by `cnn_kfold.py` for each hosts.

  [./Models] Place where trained models are stored. Note that freshly-trained models will overwrite existing ones.

  [./Results] Temporary place where predicted results are processed.

## Usage
  ### Prepare variants list for prediction.
  Modify or create the file `var_chk.csv`.
  
  See `var_chk_circ.csv` for examples.

  Note that for non-human hosts, the residue indices are 318 smaller than the expected one.
  For instance, deer_166Kcorresponds to E484K for deer, while humancov2_484K or simply 484K corresponds to E484K for human.

  ### Predict K<sub>D,app</sub> ratios for variants in `var_chk.csv` with 25 pre-trained models.
  `./cnn_kfold.py 5`
  
  ### 5-fold cross-validation test, generating 5 models.
  `./cnn_kfold.py 3` 
