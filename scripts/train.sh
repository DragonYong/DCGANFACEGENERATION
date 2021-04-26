OUTPUT=
python dcgan_face_generation.py \
    --OUTPUT=/OUTPUT/DCGAN/LFW \
    --DATASETS_NAME=lfw_new_imgs \
    --BATCH_SIZE=100 \
    --Z_DIM=100 \
    --WIDTH=64 \
    --HEIGHT=64 \
    --EPOCHS=100 \
    --LEARNING_RATE=0.0002 \
    --PER_PLOT=500
