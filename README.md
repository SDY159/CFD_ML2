# CFD_ML2
GPT-2 and CNN model for particle trajectory and erosion prediction in OP-650 boiler header

We utilized and modified the code from TrphysX: 
@article{geneva2020transformers,
    title={Transformers for Modeling Physical Systems},
    author={Geneva, Nicholas and Zabaras, Nicholas},
    journal={arXiv preprint arXiv:2010.03957},
    year={2020},
    URL={https://pypi.org/project/trphysx/}
}
    
We present a new hybrid ML method, codenamed FLUID-GPT: Fast Learning to Understand and Investigate Dynamics with a Generative Pre-Trained Transformer. 
FLUID-GPT utilizes a Generative Pre-Trained Transformer 2 (GPT-2) with a 3D CNN to predict particle trajectories and erosion in an industrial-scale boiler header. 
GPT-2 was initially developed for natural language processing/translation, which has garnered recent widespread attention for its use with interactive Artificial Intelligence (AI) chatbots or forecasting models for time series data. 
The transformer model in GPT-2 employs an encoder-decoder architecture where the encoder extracts features from an input sequence, and the decoder generates an output sequence utilizing these features with another set of input sequences.
In this work, we utilized our FLUID-GPT ML approach to predict particle trajectories based on five initial parameters (particle size, main-inlet speed, main-inlet pressure, sub-inlet speed, and sub-inlet pressure) followed by erosion predictions based on the GPT-2-learned trajectories. 
