PYT=python3
CUDA="2"
NUM_CLONES=1

MODEL="mobilenet_v1"
WIDTH=075 #025 050 075 nothing
if [ -z ${WIDTH} ]
then
	WIDTH=1
else
	MODEL+="_${WIDTH}"
fi

DATASET="visualwakewords"
DATASET_DIR="visualwakewords_vehicle"

TRAIN_DIR="${MODEL}/${DATASET_DIR}/224_${WIDTH}/"
EVAL_DIR="${TRAIN_DIR}/eval"

NUM_EPOCHS=100
BATCH_SIZE=96
LR=0.045
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
