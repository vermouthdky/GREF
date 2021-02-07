for dataset in 'cora' 'citeseer'; do
    for ptb_type in 'meta'; do
        for ptb_rate in 0 0.05 0.1 0.15 0.2 0.25; do
            python baselines.py --cuda_num 0 --dataset=$dataset --ptb_rate=$ptb_rate --ptb_type=$ptb_type --model_type='GCN_SVD'
        done
    done
done
