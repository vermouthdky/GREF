for alpha in 1.0 0.5 0.1 0.02
do
    python main.py --cuda_num=0 --type_norm='group' --train --type_model='NLGCN' --dataset='Cora' --alpha=$alpha
done