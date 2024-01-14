import os
import sys

sh_file_name = 'run_Main_pub_T1.sh'
sh_file_save_path = '/home/eckert/Codes/TextureOpt/Texture_code_v1.5'
# env_activate = True
# env_name = 'pytorch3d_2'

nGPU = None  # None, '0,1,2,3'
name_code = 'Main.py'
file_path = '/home/eckert/Datasets/texture_data/pub/scene3d_key'
file_names = os.listdir(file_path)
file_names = sorted(file_names)

fp = open(os.path.join(sh_file_save_path, sh_file_name),'w')
# if env_activate:
# 	fp.write('conda activate '+ env_name + '\n')
for f_name in file_names:
	# ours_scene3d
	# config_terms = "--initialized 1 --opti_mode 'alter_adapt' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/pub/scene3d_key/%s' --obj_dir '../../../Datasets/texture_data/pub/scene3d_key/shape' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 0.1 --depth_weight 1.0 --laplacian_weight 0.05 " + \
	# 	"--lr_G_T 1e-2 --lr_G_V 5e-5 --lr_G_C 1e-4 --batchSize 10 --num_T_unit 1 --num_V_unit 1 --num_C_unit 1 " + \
	# 	"--output_dir '../../../Results_texture_opti/TexOpti/pub/scene3d_key/%s'"%f_name
	# ATO_scene3d
	config_terms = "--initialized 1 --opti_mode 'alter_fix' --data_type 'RGBD' " + \
		"--input_dir '../../../Datasets/texture_data/pub/scene3d_key/%s' --obj_dir '../../../Datasets/texture_data/pub/scene3d_key/shape' "%f_name + \
		"--lr_G_T 1e-3 --batchSize 10 --num_T_unit 1 --num_V_unit 0 --num_C_unit 0 " + \
		"--output_dir '../../../Results_texture_opti/ATO/pub/scene3d_key/%s'"%f_name
	# ours_pub
	# config_terms = "--initialized 1 --opti_mode 'alter_adapt' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/pub/%s' --obj_dir '../../../Datasets/texture_data/pub/shape' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 0.1 --depth_weight 1.0 --laplacian_weight 0.05 " + \
	# 	"--lr_G_T 1e-2 --lr_G_V 5e-5 --lr_G_C 1e-4 --batchSize 3 --num_T_unit 1 --num_V_unit 1 --num_C_unit 1 " + \
	# 	"--output_dir '../../../Results_texture_opti/TexOpti/pub/%s'"%f_name
	# ATO_pub
	# config_terms = "--initialized 1 --opti_mode 'alter_fix' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/pub/%s' --obj_dir '../../../Datasets/texture_data/pub/shape' "%f_name + \
	# 	"--lr_G_T 1e-3 --batchSize 3 --num_T_unit 1 --num_V_unit 0 --num_C_unit 0 " + \
	# 	"--output_dir '../../../Results_texture_opti/ATO/pub/%s'"%f_name
	# ours_chairs
	# config_terms = "--TransView --initialized 1 --opti_mode 'alter_adapt' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/pub/chairs_key/%s' --obj_dir '../../../Datasets/texture_data/pub/chairs_key/shape' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 0.1 --depth_weight 1.0 --laplacian_weight 0.05 " + \
	# 	"--lr_G_T 1e-2 --lr_G_V 1e-4 --lr_G_C 1e-4 --batchSize 10 --num_T_unit 1 --num_V_unit 1 --num_C_unit 1 " + \
	# 	"--output_dir '../../../Results_texture_opti/TexOpti/pub/chairs_key/%s'"%f_name
	# ATO_chairs
	# config_terms = "--TransView --initialized 1 --opti_mode 'alter_fix' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/pub/chairs_key/%s' --obj_dir '../../../Datasets/texture_data/pub/chairs_key/shape' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 0.1 --depth_weight 1.0 --laplacian_weight 0.05 " + \
	# 	"--lr_G_T 1e-3 --batchSize 10 --num_T_unit 1 --num_V_unit 0 --num_C_unit 0 " + \
	# 	"--output_dir '../../../Results_texture_opti/ATO/pub/chairs_key/%s'"%f_name
	# ours_hw m20
	# config_terms = "--TransView --initialized 1 --opti_mode 'alter_adapt' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/HW/new_M20/%s' --obj_dir '../../../Datasets/texture_data/HW/new_M20/shape_new' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 1.0 --depth_weight 1.0 --laplacian_weight 0.1 " + \
	# 	"--lr_G_T 1e-2 --lr_G_V 5e-5 --lr_G_C 1e-4 --batchSize 13 --num_T_unit 1 --num_V_unit 1 --num_C_unit 1 " + \
	# 	"--output_dir '../../../Results_texture_opti/TexOpti/new_M20/%s'"%f_name
	# ATO_hw m20
	# config_terms = "--TransView --initialized 1 --opti_mode 'alter_fix' --data_type 'RGBD' " + \
	# 	"--input_dir '../../../Datasets/texture_data/HW/new_M20/%s' --obj_dir '../../../Datasets/texture_data/HW/new_M20/shape_new' "%f_name + \
	# 	"--l1_weight 0.1 --IOU_weight 1.0 --depth_weight 1.0 --laplacian_weight 0.1 " + \
	# 	"--lr_G_T 1e-3 --batchSize 13 --num_T_unit 1 --num_V_unit 0 --num_C_unit 0 " + \
	# 	"--output_dir '../../../Results_texture_opti/ATO/new_M20/%s'"%f_name

	if nGPU is not None:
		fp.write('CUDA_VISIBLE_DEVICES='+ nGPU +' python ' + name_code + ' ' + config_terms + '\n')
	else:
		fp.write('python ' + name_code + ' ' + config_terms + '\n')
fp.close()


