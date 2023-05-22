# CFD_ML2
GPT-2 and CNN model for particle trajectory and erosion prediction in OP-650 boiler header

We utilized and modified the code from TrphysX: 
@article{geneva2020transformers,
    title={Transformers for Modeling Physical Systems},
    author={Geneva, Nicholas and Zabaras, Nicholas},
    journal={arXiv preprint arXiv:2010.03957},
    year={2020}    
}
URL: https://pypi.org/project/trphysx/

We present a new hybrid ML method, codenamed FLUID-GPT: Fast Learning to Understand and Investigate Dynamics with a Generative Pre-Trained Transformer. 
FLUID-GPT utilizes a Generative Pre-Trained Transformer 2 (GPT-2) with a 3D CNN to predict particle trajectories and erosion in an industrial-scale boiler header. 
GPT-2 was initially developed for natural language processing/translation, which has garnered recent widespread attention for its use with interactive Artificial Intelligence (AI) chatbots or forecasting models for time series data. 
In this work, we utilized our FLUID-GPT ML approach to predict particle trajectories based on five initial parameters (particle size, main-inlet speed, main-inlet pressure, sub-inlet speed, and sub-inlet pressure) followed by erosion predictions based on the GPT-2-learned trajectories. 


# Description of Python Files

1. CFD_write_hdf5_consistent_kfold.py
    It loads CFD data (Particle trajectories, 5 initial patameters, and erosion) and make them as hdf5 dataset with the K-fold cross validation. 

2. CNN_SDY159.py
    CNN model architecture (includes visualiation of erosion)
    
3. GPT-2_SDY159.py
    GPT-2 model architecture (includes visualiation of trajectory)
 
4. GPT_to_CNN_write_hdf5_consistent_kfold.py
    The predicted trajectory from GPT-2 model will be saved in the same K-fold cross validation as "CFD_write_hdf5_consistent_kfold.py" file.
    
5. LSTM_to_CNN_write_hdf5_consistent_kfold.py
    The predicted trajectory from LSTM/BiLSTM model will be saved in the same K-fold cross validation as "CFD_write_hdf5_consistent_kfold.py" file.
