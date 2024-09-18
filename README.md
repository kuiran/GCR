## Installation
Our code is based on MMtracking master branch [Link](https://github.com/open-mmlab/mmtracking), you can fllow the installation guiding [installation](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md) to create your environment.
## Model
We provide our pretrain model on [model_link](https://drive.google.com/file/d/1GxfBa8HSMPbZ3BpeBBQIaorqsPz-iQSw/view?usp=drive_link)
## Training & Inference
### Training
    bash ./tools/dist_train.sh ${config_path} 8 --work-dir ${dir}
    bash ./tools/dist_test.sh ${config_path} 8 \
     --checkpoint {model_path} \
     --out {work_dir}/result.pkl \
     --work-dir {work_dir} \
 
### Inference
  #### conver pkl to txt
     python ./utils/pkl2txt.py \
     --pkl_path {work_dir}/result.pkl \
     --txt_save_path {work_dir}/ \
     --txt_name result.txt
  
     bash ./tools/dist_test.sh {config_path} 8 \
     --checkpoint {model_path} \
     --out {work_dir}/point_results.pkl \
     --eval track\
     --work-dir ${work_dir}stark_st1_r50_500e_lasot \
     --cfg-options data.test.replace_first_frame_ann=True \
     data.test.first_frame_ann_path={work_dir}/result.txt
