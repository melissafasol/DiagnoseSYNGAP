# From Data to Diagnosis: Explainable AI techniques in *SYNGAP1* Biomarker Discovery

Mutations in the SYNGAP1 gene are a major genetic risk factor for neurodevelopmental disorders (NDDs). A large number of
genes involved in synaptic function have been implicated in rare de novo NDDs. 

We calculate three categories of features to assess synaptic function from EEG recordings: 
(1) Spectral features 
(2) Connectivity features
(3) Complexity features

These features are used to train an XGBoost and LightGBM classifier to distinguish SYNGAP1 characteristics from controls, and apply explainable artificial intelligence (XAI) techniques to identify the most influential features in genotype prediction. 

