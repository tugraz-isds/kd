# BERT-ConvE

In this project context-aware BERT embeddings are extracted and are used to replace the embedding layer of ConvE, to achieve text-aware knowledge graph embedding model.


## Folder structure

```
nlp_repo
└── ConvE
    └── data
        ├── atomic
        └── conceptnet100k

bertconve
├── data
│   └── processed
│       ├── atomic-id
│       └── conceptnet100k-id
├── README.md
├── requirements.txt
├── bertconve
└── scripts

```

## Downloading data

For experiments with Conceptnet100k, dataset can be downloaded from [here](https://ttic.uchicago.edu/~kgimpel/commonsense.html). 


For experiments with ATOMIX, dataset can be downloaded from [here](https://homes.cs.washington.edu/~msap/atomic/). 

For experiments with FB15k-237 (incl. node text attributes), dataset can be downloaded from [here](https://github.com/yao8839836/kg-bert)


## Preprocessing

For preprocessing data, the above downloaded data need to be saved under `../nlp_repo/ConvE/data/${DATASET_NAME}`, before running script `path/to/python scripts/preprocess.py ${DATASET_NAME}`

## Fine-tuning

For fine-tuning BERT model, we use the [example script from Hugging Face](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py)

```
LM_NAME=bert-base-uncased
NNB=1
TRAIN_FILE=train_${NNB}neighbor_sent.txt
VALID_FILE=valid_${NNB}neighbor_sent.txt

CUDA_VISIBLE_DEVICES=0,1 \
path/to/python run_mlm.py \
--model_name_or_path $LM_NAME \
--train_file path/to/bertconve/data/processed/${DATASET_NAME}-neighbor/${TRAIN_FILE} \
--validation_file path/to/bertconve/data/processed/${DATASET_NAME}-neighbor/${VALID_FILE} \
--do_train \
--do_eval \
--output_dir $OUTPUT_DIR \
--line_by_line \
--num_train_epochs 15 \
--per_device_train_batch_size 90 \
--learning_rate 5e-5 \
```

## Feature extraction

For feature extraction, see script `scripts/feature_extract.py`. Example usage: `path/to/python scripts/feature_extract.py bert-base-uncased avg from_5nb ${DATASET_NAME}`

## ConvE

For ConvE model, we updated the [original repository](https://github.com/TimDettmers/ConvE) to include BERT-embeddings. The updated version see `../nlp_repo/ConvE`. Example usage:

```
CUDA_VISIBLE_DEVICES=2 \
path/to/python main.py \
--model bertconve \
--data $DATASET_NAME \
--hidden-drop 0.3 \
--input-drop 0.2 \
--feat-drop 0.2 \
--lr 0.001 \
--batch-size 128 \
--label-smoothing 0.1 \
--fn-bert-embedding path/to/bertconve/output/${EMB_FILE_NAME} \
--embedding-dim 768 \
--embedding-shape1 32 \
--conv-channel 32 \
--kernel-size 3 \
--hidden-size 43648 \
```
