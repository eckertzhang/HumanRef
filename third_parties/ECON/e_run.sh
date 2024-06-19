python -m third_parties.ECON.apps.infer -cfg ./configs/econ.yaml -in_dir ./images -out_dir ./results/human_test2
python third_parties/ECON/run_ECON.py --in_dir='./data/image_000355.jpg' --out_dir='./data/Results_ECON'

python third_parties/ECON/run_ECON_3d.py --in_dir='/apdcephfs/share_1330077/eckertzhang/Dataset/cape_test_data/cape_3views' --out_dir='/apdcephfs/share_1330077/eckertzhang/Dataset/cape_test_data/results_econ' --dataset_type='cape'
python third_parties/ECON/run_ECON_3d.py --in_dir='/apdcephfs/share_1330077/eckertzhang/Dataset/thuman2_icon/thuman2_36views0' --out_dir='/apdcephfs/share_1330077/eckertzhang/Dataset/thuman2_icon/results_econ' --dataset_type='thuman2'