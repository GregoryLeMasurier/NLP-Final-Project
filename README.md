# How to use:  
Train a full model: python full_model/full_model.py  
Train a base model: python random_model/random_model.py  
In demo.ipynb, replace the config name (pretrained_config.json) to the config you would like to use.  
In demo.ipynb, replace the model name (pretrained_model.bin) to the model you would like to use.  
Run the block in demo.ipynb to run the demo using models you train  

# Dependencies:  
python -m pip install jupyterlab torch transformers datasets scikit-learn ipywidgets  
conda install -c anaconda cudatoolkit  
conda list | grep cudatoolkit  
Use the version number from the above to replace ###: pip install bitsandbytes-cuda###  
pip install numba  
pip install gradio  
pip install rouge-score nltk sentencepiece  
