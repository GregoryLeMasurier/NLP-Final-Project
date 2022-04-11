# Homework 5. Transformer machine translation model

> Do not modify code outside `# YOUR CODE STARTS HERE` unless explicitly allowed by a TA. Failure to follow this will lead to failing the homework.

In this homework, we will finish Transformer implementation and train a machine translation model.

We have tested this code with a cleaned version of the WMT-14 English-German dataset (`stas/wmt14-en-de-pre-processed`).
It is a medium-size dataset that contains 802.743 sentence pairs which are enough to train a decent translation system.

> If you want to use a different language pair, please ask TA for a dataset recommendation (it can be hard to find a reasonably sized dataset, 10Mb is not enough and 20Gb is just hard to work with). If you want to use your own dataset, inform TA about it too.

## 1. Implement Cross-Attention, Decoder Layer, and Transformer Encoder-Decoder Model
Start with `notebooks/01_transformer_decoder.ipynb`. It contains guides and instructions to implementing cross-attention within a multi-head attention class. You will need to implement the rest of the Transformer architecture, including `TransformerDecoderLayer` and `TransformerEncoderDecoderModel`.Your implementation should pass all of the tests.

There are 8 coding tasks in this part.

## 2. Train tokenizers for your dataset

Go to `cli/create_tokenizer` and complete the script. Then train a tokenizer. 

Command example:

```bash
python cli/create_tokenizer.py \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --dataset_config ende \
    --vocab_size 32_000 \
    --save_dir en_de_output_dir \
    --source_lang en \
    --target_lang de

```

Because the dataset is large, this can take 5 minutes or more. Start worrying if it does not finish after 1 hour.

## 3. Train a Transformer Machine Translation Model

Open `notebooks/02_get_familiar_with_train.ipynb` and read it carefully. It will guide you through the script and help you to solve 3 coding tasks and 3 inline questions.

After finishing all of the tasks is time to train a model.

In the beginning, try out a very small (2 layers) model in `--debug` mode to make sure the script can finish without failing. Then, find the largest batch size your GPU memory can fit and the largest learning rate that does not cause your model to diverge. Also feel free to play with any other hyperparameters, such as model size learning rate schedule, regularization, and so on.

When selecting the number of epochs, remember that the dataset is big. 1 epoch can take 6 hours or even more even on capable hardware. However, because the dataset is so large, 1 epoch might be enough to get a model that performs ok(-ish) with a BLEU score of around 20. Alternatively to the number of epochs, you can provide `--max_train_steps`.

Here's an example script that works ok(-ish)

```
python cli/train.py \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --dataset_config ende \
    --source_lang en \
    --target_lang de \
    --output_dir en_de_output_dir \
    --batch_size 64 \
    --num_warmup_steps 5000 \
    --learning_rate 3e-4 \
    --num_train_epochs 1 \
    --eval_every 5000
```

## Try out your model

After training, use `notebooks/03_interact.ipynb` to translate texts using your model. This part is evaluated and (we hope) might be rewarding, as the final system should produce reasonable translations, please do not skip it.

## Hyperparameters

Feel free to change hyperparameters, but remember that:
* It takes hours to train a single model using a GPU.
  * Meaning you need to start training at least a day or two before the deadline
  * Meaning you won't be able to play with hyperparameters a lot
  * Meaning you can't use your laptop (Core i9 laptop needs more than 100hours to train the model)
  * Meaning you need to learn how to work with a GPU server.
* Smaller models are faster to train until convergence, but larger models sometimes can reach the same loss in fewer steps and less time.
* Try to maximize your batch size and fill all of the GPU memory.
  * Batch size should be at least 8 and preferably around 64 or even more.
  * If you see an out-of-memory error, probably your batch size or max_length or your network parameters are too big. Do not reduce max_length beyond 128.
  * To reduce memory consumption, you can use Adafactor optimizer instead of Adam. Here is the [documentation on how to use it](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor)
* Keep all of your shapes and sizes divisible by 8, this makes the GPUs a bit more efficient.
* You can use an empirical [scaling law](https://arxiv.org/abs/2001.08361) to estimate your learning rate `lr = 0.003239 - 0.0001395 log(N)`. Here, N is the number of parameters of your model, excluding embeddings (should be around 1-100M if you are using something like a small or base transformer).
* Your final model will be evaluated based on its BLEU score.

Finally, run `cli/train.py` and provide your selected hyperparameters to it. Save your model to `output_dir`.

**Monitor your model performance while it is training** in WadnB. This is literally the main purpose of this tool. If the model is not improving at all or is diverging, look up our "my model does not converge" checklist from Lecture 3. At the same time, if you get a very high test accuracy, your model might be cheating and your causal masking or data preprocessing is not implemented correctly. To help you understand what the correct training loss plot and eval perplexity/accuracy should look like, we will post a couple of images in Slack. Your runs will most probably look different, because of different hyperparameters, but they should not be extremely different.

> To interrupt the script, press CTRL+C

> You can download your model and tokenizer from the server to your laptop using `scp` command-line tool ([how to use it](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/)).


## Connecting to Google Cloud

Please use Google Cloud for this homework.

Follow this Stanford tutorial on how to connect to Google Cloud. Feel free to ask questions about google cloud in #discussion channel in Slack (but do not post your passwords or access tokens there or anywhere else publically, including your github).

> If you are unfamiliar with SSH, look up this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server) and if you are using Windows, [this one (Windows 10 native SSH)](https://www.howtogeek.com/336775/how-to-enable-and-use-windows-10s-built-in-ssh-commands/) or [this one (PUTTY)](https://mediatemple.net/community/products/dv/204404604/using-ssh-in-putty-(windows)).

## How To Keep Running Commands After SSH Session Disconnection

It takes hours to run training on a GPU. However, if you would just run `python cli/train.py <blahblah>` in your SSH session and then disconnect, your script will automatically be shutdown by the system. To avoid this, you can use [tmux sessions](https://leimao.github.io/blog/Tmux-Tutorial/), you can also use `screen`, if you are familiar with it, but **do not** use `nohup` as it is not flexible enough for our purposes.

### How to sync your laptop code and server code

Use Github and Git for that. These are essential professional instruments and you will need to learn them at some point anyway. Create a **private** repository for your homework (your grade will be lowered, if the repository is public). And use git commands to synchronize your code. You will mostly only need these ones: `git commit`, `git push`, `git clone`, and `git pull` .

> If you are unfamiliar with Git and Github: [read this tutorial](https://docs.github.com/en/get-started/using-git/about-git).

> Git and GitHub are extremely popular. If you see an error — google it! You will find the answer way quicker than contacting a TA. But feel free to ask you questions in Slack too, especially if you do not understand the answer from Google.

If you have trouble with understanding git, please contact a TA. You can use `scp` command to sync your code but **it can cause losing your changes** if you are not careful. We advise **not** to use `scp` to sync your code, but only to use it for large file upload/download.

## Setting up the environment

We strongly recommend using a code editor like VSCode or PyCharm. Jupyter Lab or Spyder are jupyter-notebook-first tools and are not well-suited to work with regular python modules. Here is a good tutorial on how to [setup VSCode for Python](https://www.youtube.com/watch?v=Z3i04RoI9Fk). Both of them also support jupyter notebooks, you just need to specify which jupyter kernel you want to use (most probably its `nlp_class`). For VSCode you may want to additionally install a [markdown extention](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) to render files like this README.md.

You will be developing the package that is located inside `transformer_mt`. In order to be able to import this package from the notebooks, training script, and anywhere else, you need to 

> Note that even though this homework is similar to language modeling homework, the package name is different. It is `transformer_mt` instead of `transformer_lm`. If you see any references to `transformer_lm` contact a TA so that they would fix it.

1. If you are working in a new environment (for example, a server), you need to create a new conda environment (`conda create -n nlp_class python=3.7`).
2. Activate your python environment (e.g., `conda activate nlp_class`).
3. Go to the homework directory that contains `setup.py` file (the same directory this `README.md` is in).
4. Install the package using the command `pip install --editable .`. It should download all of the dependencies and install your module.
5. If you are on a GPU machine, you need to install a GPU version of PyTorch. To do that, first, check what CUDA version your server has with `nvidia-smi`.
   1. If your CUDA version is below 10.2, don't use this server
   2. If your CUDA version is below 11, run `pip install torch`
   3. If your CUDA version is 11.X run, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
   4. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above commands (usually this helps), if it doesn't help, describe your problem in `#discussion`.
   5. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above. 

**Explaining pip install -e . command**:
`pip` is the python package manager. `install` is the pip command to install the module. `-e` is the argument that says to install the module in *editable* mode which allows you to edit the package and import from it right away without the need to reinstall it. The final argument is `.` which says "the package is located in the current directory".

# Submitting this homework

> NOTE: Do not add `model.pt` and other large files to your git repository.

1. Restart your `.ipynb` notebooks and execute them top-to-bottom via the "Restart and run all" button.
Not executed notebooks or the notebooks with the cells not executed in order will receive 0 points.
1. **If you are using GitHub**, add `github.com/guitaricet` and `github.com/NamrataRShivagunde` to your [repository collaborators](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository), so we could access your code. Push your last changes (including the executed notebooks) and submit the link to your GitHub repository to the Blackboard.
2. **If you are not using GitHub**, delete `output_dir` (or move its contents somewhere else if you want to reuse them later) and `wandb` directories. Zip this directory and submit the archive to the Blackboard.
3. Submit a link to your best wandb run to the Blackboard too. You will be evaluated based on the BLEU score of your model. Make sure your wandb project is public.
