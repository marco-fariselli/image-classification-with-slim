PYT=python3
CUDA="2"
NUM_CLONES=1

while getopts w:m:e:b:l: option
do
	case "${option}"
	in
	w) WIDTH=${OPTARG};;
	m) MODEL=${OPTARG};;
	b) BATCH_SIZE=${OPTARG};;
	e) NUM_EPOCHS=${OPTARG};;
	l) LR=${OPTARG};;
	esac
done

MODEL=${MODEL:="mobilenet_v1"}
WIDTH=${WIDTH:=}
if [ -z ${WIDTH} ]
then
	WIDTH=1
else
	MODEL+="_${WIDTH}"
fi

DATASET="visualwakewords"
DATASET_DIR="visualwakewords_vehicle"

TRAIN_DIR="${MODEL}/${DATASET_DIR}/train_dir"
EVAL_DIR="${TRAIN_DIR}/eval"

NUM_EPOCHS=${NUM_EPOCHS:=100}
BATCH_SIZE=${BATCH_SIZE:=96}
LR=${LR:=0.045}
DS_TRAIN_DIM=82783

NUM_STEPS=0

DATASET_FLAGS="--dataset_name=${DATASET} --dataset_dir=${DATASET_DIR}"
TRAIN_FLAGS="--learning_rate=${LR} --batch_size=${BATCH_SIZE} --num_clones=${NUM_CLONES}"

for ((i=1; i<=${NUM_EPOCHS}; i++))
do
	let NUM_STEPS=$((i*${DS_TRAIN_DIM}/${BATCH_SIZE}))
	#echo ${TRAIN_FLAGS}
	CUDA_VISIBLE_DEVICES=${CUDA} ${PYT} train_image_classifier.py --train_dir=${TRAIN_DIR} ${TRAIN_FLAGS} --max_number_of_steps=${NUM_STEPS} ${DATASET_FLAGS} --dataset_split_name="train"  --model_name=${MODEL} --use_grayscale --preprocessing_name=mobilenet_v1 
	CUDA_VISIBLE_DEVICES=${CUDA} ${PYT} eval_image_classifier.py --eval_dir=${EVAL_DIR} --checkpoint_path=${TRAIN_DIR} ${DATASET_FLAGS} --dataset_split_name="val" --model_name=${MODEL} --use_grayscale --preprocessing_name=mobilenet_v1
done

${PYT} export_inference_graph.py --model_name=${MODEL} --output_file="${MODEL}/${MODEL}_inference.pb" --input_size=224 --use_grayscale --dataset_name=${DATASET}

#freeze_graph --input_graph="${MODEL}/${MODEL}_inference.pb" --output_graph="${MODEL}/${DATASET_DIR}/frozen_graph.pb" --input_binary --input_checkpoint=${TRAIN_DIR} --output_node_names="MobilenetV1/Predictions/Reshape_1"

#tflite_convert --graph_def_file="${MODEL}/${DATASET_DIR}/frozen_graph.pb" --output_file="${MODEL}/${DATASET_DIR}/${MODEL}_grayscale_vwwvehicle.tflite" --input_arrays="input" --output_arrays="MobilenetV1/Predictions/Reshape_1"
