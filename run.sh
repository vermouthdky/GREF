for alpha in 1.0; do
    python main.py --cuda_num=3 --type_model='NLGCN' --dataset='Cora' --alpha=$alpha --ks 0.9 0.8 0.7
done
