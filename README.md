# Group Members:
Gregory LeMasurier and Mojtaba Talaei Khoei

# Project Description:
In this project we compare a randomly initialized PEGASUS model to a pretrained PEGASUS model for abstractive summarization. We trained and fine-tuned our models on the cnn_dailymail dataset. We found that the pretrained model performed much better than our randomly initialized model, as seen by the Rouge scores.

# DEPENDENCY NOTE:  
Install all dependencies in the requirements.txt.  
NOTE: Please read the instructions to install bitsandbytes
properly as it depends on the cudatoolkit version!  

# Train the models:
Train a base model: python random_model/random_model.py  
Train a pretrained model: python full_model/full_model.py  

# Run the demo:
Modify cell two to include the proper path and file name to your model config and bin file.  
Run all cells in demo.ipynb to run the demo using model you trained.  
Go to the local url output by the 3rd block, for us it was: http://127.0.0.1:7860/  
Paste text you would like to sumarize in the left most gradio block.  
Press submit and the summary will be displayed in the rightmost block.  

# WANDB Results:
https://wandb.ai/glemasurier/PegasusSummarization?workspace=user-glemasurier
