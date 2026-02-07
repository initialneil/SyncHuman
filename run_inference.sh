CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch --main_process_port 9061 \
    inference_OneStage_guava.py \
    --data_root /home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v1 \
    --output_root /home/szj/err_empty_syn/ytb/bbox_square_x1.1/syn_human \
    --reverse_order \

