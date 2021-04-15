# Term paper project on "Text summarization using pretrained encoders"
This repository contains the code of our project done on the EMNLP 2019 paper [**Text Summarization with Pretrained Encoders**](https://arxiv.org/abs/1908.08345) as a part of the course [BITS F312] Neural Networks and Fuzzy Logic at BITS Pilani.

The original repository for the paper can be found [here](https://github.com/nlpyang/PreSumm).

## Directory Structure
```bash
.
├── Custom_Dataset_BBC.ipynb
├── LICENSE
├── Model_Training_and_Graph_Plotting.ipynb
├── README.md
├── Rouge_Score_Evaluation.ipynb
├── Summary_Generation.ipynb
├── bert_data
├── custom_data_training
│   └── data_builder.py
├── json_data
├── logs
├── models
├── raw_data
├── requirements.txt
├── results
├── src
│   ├── cal_rouge.py
│   ├── distributed.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── adam.py
│   │   ├── data_loader.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   ├── loss.py
│   │   ├── model_builder.py
│   │   ├── neural.py
│   │   ├── optimizers.py
│   │   ├── predictor.py
│   │   ├── reporter.py
│   │   ├── reporter_ext.py
│   │   ├── trainer.py
│   │   └── trainer_ext.py
│   ├── others
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── pyrouge.py
│   │   ├── tokenization.py
│   │   └── utils.py
│   ├── post_stats.py
│   ├── prepro
│   │   ├── __init__.py
│   │   ├── data_builder.py
│   │   ├── smart_common_words.txt
│   │   └── utils.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_abstractive.py
│   ├── train_extractive.py
│   └── translate
│       ├── __init__.py
│       ├── beam.py
│       └── penalties.py
└── urls
```

## Code files

### Inside src

[train.py](src/train.py) - Contains the main training workflow of the models. 

[train_abstractive.py](src/train_abstractive.py) - Contains the training, validation and testing workflow of the abstractive model in a distributed manner. 

[train_extractive.py](src/train_extractive.py) - Contains the training, validation and testing workflow of the extractive model in a distributed manner. 

[preprocess.py](src/preprocess.py) - Wraps around functions defined in [databuilder.py](src/prepo/databuilder.py) file and provides a workflow for preprocessing the dataset from raw input files into a form that can be fed to the model.

[distributed.py](src/distributed.py) - Contains helper functions to support distributed training using multiple GPUs.

[cal_rouge.py](src/cal_rouge.py) - To calculate the rouge scores of the generated summaries. 

[post_stats.py](src/post_stats.py) 

### Inside src/models
[adam.py](src/train/adam.py) - Implements the adam optimizer algorithm.

[data_loader.py](src/train/data_loader.py) - Loads the dataset and iteratively passes them in batches.

[decoder.py](src/train/decoder.py) - Defines the structure of the Transformer decoder network with attention mechanism.

[encoder.py](src/train/encoder.py) - Defines the structure of the Transformer encoder network.

[loss.py](src/train/loss.py) - Handles the details of loss computation during training.

[model_builder.py](src/train/model_builder.py) - Integrates the enocoder and decoder architecture, along with the optimizer to define the model.

[neural.py](src/train/neural.py) - Contains implementation of feedforward and multi-head attention layers.

[optimizers.py](src/train/optimizers.py) - Contains controller class for optimization and function to update the model parameters based on current gradients.

[predictor.py](src/train/predictor.py) - Translates the generated output and gives the predicted summary.

[reporter_ext.py](src/train/reporter_ext.py) and [reporter.py](src/train/reporter.py) - Functionalities to report metrics during the training steps.

[trainer_ext.py](src/train/trainer_ext.py) and [trainer.py](src/train/trainer.py) - Includes functions that controls the training process and defines the workflow.
 

## Tasks assigned
1. Train the models and plot the relevant metrics(loss/F1/accuracy/etc) with respect to epochs.
2. Compute and report the Rouge scores for your trained model.
3. Select 3-4 example articles and generate their summaries using your trained model - both extractive and abstractive.
4. Use a custom summarization dataset of your choice, train the model on this data and report your findings.

## Instruction to run the tasks
Optionally, create a virtual environment on your system and open it. 

To run the application, first clone the repository by typing the command in git bash.
```
git clone https://github.com/AnushkaDayal/PreSumm_NNFL.git
```

Alternatively, you can download the code as .zip and extract the files.

Shift to the cloned directory
```
cd PreSumm_NNFL
```

To install the requirements, run the following command:
```
pip install -r requirements.txt
```

## Data Preparation for CNN/Dailymail
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)

unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Task-1 
Task-1 includes training the model on CNN/DailyMail data and plotting the relevant graphs. Follow the steps given in the ```Model_Training_and_Graph_Plotting.ipynb``` file to accomplish this task.

## Task-2
Task-2 includes calculating the ROUGE scores on the test dataset from our trained model. You can download our custom trained models from the **Pre-trained Models** section below. Follow the steps in ```Rouge_Score_Evaluation.ipynb``` to accomplish this task.

The results we obtained on the CNN/DM testing dataset were as follows:  -

<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (CNN/DM) </td>
    <td class="tg-0pky">42.37</td>
    <td class="tg-0pky">19.59</td>
    <td class="tg-0pky">38.76</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (BBC)</td>
    <td class="tg-0pky">35.96</td>
    <td class="tg-0pky">13.79</td>
    <td class="tg-0pky">32.42</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExtAbs (CNN/DM)</td>
    <td class="tg-0pky">30.65</td>
    <td class="tg-0pky">10.98</td>
    <td class="tg-0pky">28.86</td>
  </tr>
</table>

## Task-3
Task-3 includes generating summaries on raw input. We have provided the raw input we used in the ```raw_data``` folder. Follow the steps mentioned in ```Summary_Generation.ipynb``` to generate the summaries on the raw input.
For abstractive purposes: Each line in your input raw text file must be a single document
For extractive purposes: You must insert [CLS] [SEP] as your sentence boundaries.

## Task-4
Task-4 is about training the dataset on a custom dataset. We chose the **BBC Extractive dataset** for this purpose. The dataset can be downloaded from [here](https://www.kaggle.com/pariza/bbc-news-summary/data). Our custom trained model can be found in the **Pre-trained Models** section below. The steps for the training are mentioned in the ```Custom_Dataset_BBC.ipynb``` file. 
To be able to follow the preprocessing steps above for your custom dataset, replace the "data_builder.py" file found in /src/prepro directory by the one in custom_data_training/

## Pretrained Models
[Custom trained BertSumExt on CNN/DM dataset](https://drive.google.com/file/d/1rJaH1hEFWrz05xW4QHS1Kf5dobcHKJIZ/view?usp=sharing)

[Custom trained BertSumExtAbs on CNN/DM dataset](https://drive.google.com/file/d/1-5_TZyvWbU_C-Eac41rFRdjanKwvDC5p/view?usp=sharing)

[Custom trained BERTSUMEXT on BBC dataset](https://drive.google.com/file/d/1-VXBEka-5dzKgVxWcoh25oFQruBCyxwU/view?usp=sharing)

## Team Members
1. [Abhimanyu Sethi](https://github.com/gollum-98)
2. [Anushka Dayal](https://github.com/AnushkaDayal)
3. [Shrey Shah](https://github.com/imshreyshah)
