import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from data_engine import DataEngine





if __name__ == "__main__":


    # device="cuda:1"
    # de=DataEngine(device=device)
    # cache_path="/root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.merged.cache"
    # text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
    # de.load_cached_label(cache_path=cache_path, 
    #                     data_style="grounding", 
    #                     text_embed_pt=text_embed_pt)

    im_index=0


    de=DataEngine(device="cuda")
    de=DataEngine(device="cuda")
    yaml_config="/root/ultra_louis_work/datasets/Objects365v1.yaml"
    cache_path="/root/ultra_louis_work/datasets/Objects365v1/labels/train.cache"
    de.load_cached_label(cache_path=cache_path, data_style="detection", yaml_config=yaml_config)
    de.print_data_info()

    # de.visual_and_save2(im_index, save_path="./visualized_grounding_example.jpg")
    de=DataEngine(device="cuda")
    yaml_config="/root/ultra_louis_work/datasets/Objects365v1.yaml"
    cache_path="/root/ultra_louis_work/datasets/Objects365v1/labels/train.updated.cache"
    de.load_cached_label(cache_path=cache_path, data_style="detection", yaml_config=yaml_config)

    de.print_data_info()