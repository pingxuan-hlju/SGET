<<<<<<< HEAD
# SGET

## Introduction  
The project  is an implementation of a Learning shared topology and edge-aware representations across multiple graphs for drug–microbe association prediction.

---

## Catalogs  
- **/data_DM**: Contains the dataset used in our method.
- **/code**: Contains the code implementation of  SGET algorithm.
- **dataloader.py**: Processes the drug and microbial similarities, associations, embeddings, and adjacency matrices.
- **sim.py**: Calculates the drug attribute similarity based on the heat kernel and the microbe similarity based on the Gaussian kernel.
- **model.py**: Train and test the model.

---

## Environment  
The SGET code has been implemented and tested in the following development environment: 

- Python == 3.8,10 
- Matplotlib == 3.7.1
- PyTorch == 2.0.0  
- NumPy == 1.24.1
- Scikit-learn == 1.3.0

---

## Dataset  
- **drug_names.txt**: Contains the names of 1373 drugs.  
- **microbe_names.txt**: Contains the names of 173 microbes.
- **drugsimilarity.zip**: A compressed file that contains the following two files.
  - **drugfusimilarity.txt**: Includes the functional similarities among the drugs.
  - **newdrugsimilarity.txt**: Includes the drug  functional similarities .
- **microbe_microbe_similarity.txt**: Contains the cosine similarity of microbes.
- **newmicrobesimilarity.txt**: Contains the Gaussian kernel similarity of microbes.
- **net1.mat**: Represents the adjacency matrix of the drug-microbe heterogeneous graph.
- **Supplementary_file_SF2.xlsx**: Lists the top 20 candidate microbes for each drug.

---

## How to Run the Code  
1. **Data preprocessing**: Constructs the adjacency matrices, embeddings, and other inputs for training the model.  
    ```bash
    python dataloader.py
    ```

2. **Train and test the model**.  
    
    ```bash
    python model.py
    ```
