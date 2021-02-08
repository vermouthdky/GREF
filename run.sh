# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Cora' --metric='attention'
# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Cora' --metric='identity'
# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Citeseer' --metric='attention'
# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Citeseer' --metric='identity'
# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Pubmed' --metric='attention'
# python main.py --cuda_num 2 --type_model='NLGCN' --dataset='Pubmed' --metric='identity'

for dataset in 'CoauthorCS' 'Cora' 'Citeseer' 'Pubmed'; do
    for num_layers in 2 4 8 16 32; do
        for type_model in 'SGC' 'APPNP' 'JKNet'; do
            python main.py --cuda_num 2 --type_model=$type_model --dataset=$dataset --num_layers=$num_layers --epochs=200
        done
    done
    python main.py --cuda_num 2 --type_model='g_U_Net' --dataset=$dataset --epochs=200
    python main.py --cuda_num 2 --type_model='GCN' --dataset=$dataset --epochs=200
    python main.py --cuda_num 2 --type_model='GAT' --dataset=$dataset --epochs=200
done

# for dataset in 'Cora' 'Citeseer' 'Pubmed'; do
#     python main.py --cuda_num 2 --type_model='GCN' --dataset=$dataset
#     python main.py --cuda_num 2 --type_model='GAT' --dataset=$dataset
# done

# random attack
# for dataset in 'Cora' 'Citeseer'; do
#     for metric in 'attention' 'identity'; do
#         for ptb_rate in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
#             python main.py --cuda_num 2 --type_model='NLGCN' --dataset=$dataset --metric=$metric --ptb_rate=$ptb_rate
#         done
#     done
# done

# ablation study
# for dataset in 'Cora' 'Citeseer'; do
#     for metric in 'identity'; do
#         for ptb_rate in 0.3 0.35 0.4; do
#             python main.py --cuda_num 2 --type_model='NLGCN' --dataset=$dataset --metric=$metric --ptb_rate=$ptb_rate --alpha=0
#             python main.py --cuda_num 2 --type_model='NLGCN' --dataset=$dataset --metric=$metric --ptb_rate=$ptb_rate --gamma=0 --lamb=0
#         done
#     done
# done

# parameters study
# for ptb_rate in 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
#     for alpha in 0 0.5 1 2 4 8; do
#         python main.py --cuda_num=2 --type_model='NLGCN' --dataset='Cora' --alpha=$alpha --ptb_rate=$ptb_rate --altered_param='alpha'
#     done
#     for lamb in 0 5e-5 5e-4 5e-3 5e-2 5e-1; do
#         python main.py --cuda_num=2 --type_model='NLGCN' --dataset='Cora' --lamb=$lamb --ptb_rate=$ptb_rate --altered_param='lamb'
#     done
#     for gamma in 0 1e-5 1e-4 1e-3 1e-2 1e-1; do
#         python main.py --cuda_num=2 --type_model='NLGCN' --dataset='Cora' --gamma=$gamma --ptb_rate=$ptb_rate --altered_param='gamma'
#     done
# done
