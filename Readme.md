# Two-Space Graph Convolutional Neural Networks (TSGCNN)

TSGCNN code.

## Requirements

- python==3.7.12
- pytorch==1.10.1
- numpy==1.21.5
- pandas==1.3.5
- Scimitar-learn==1.0.2
- pubchenpy==1.0.4

## GDSC

- model.py: Code that implements the model.
- utils.py: The code that implements the tool.
- sampler.py: The code that implements the sampler.
- Directory New implements single-row and single-column zeroing experimental code.
- The directory Random implements the random zeroing experimental code.
- The directory Single implements the single drug experiment code.
- The directory Target implements the target drug experiment code.
- The directory processed_data contains the data required for the experiment.
  - cell_drg.csv records the log IC50 association matrix of cell line-drug. 
  - cell_drugbinary.csv records the binary cell line-drug association matrix. 
  - cellcna.csv records the CNA features of the cell line. 
  - cell_gene.csv records cell line gene expression features. 
  - cell_mutation.csv records somatic mutation features of cell lines. 
  - drug_feature.csv records the fingerprint features of drugs. 
  - null_mask.csv records the null values in the cell line-drug association matrix. 
  - threshold.csv records the drug sensitivity threshold.



## CCLE

Same as directory GDSC.

- CCLE/processed_data/ 
  - cell_drug.csv records the log IC50 association matrix of cell line-drug. 
  - cell_drug_binary.csv records the binary cell line-drug association matrix. 
  - cell_cna.csv records the CNA features of the cell line. 
  - drug_feature.csv records the fingerprint features of drugs. 
  - cell_gene.csv records cell line gene expression features. 
  - cell_mutation.csv records somatic mutation features of cell lines.

## Contact

If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me ([weipeng1980@gmail.com](mailto:weipeng1980@gmail.com)).