for alpha in 1.0
do
    python main.py --cuda_num=0 --train --type_model='NLGCN' --dataset='Cora' --alpha=$alpha
done