import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from data_engine import DataEngine
import numpy as np




if __name__ == "__main__":


    # device="cuda:1"
    # de=DataEngine(device=device)
    # cache_path="/root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.merged.cache"
    # text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
    # de.load_cached_label(cache_path=cache_path, 
    #                     data_style="grounding", 
    #                     text_embed_pt=text_embed_pt)

    im_index=0


    # de=DataEngine()
    # cache_path="/root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.merged.cache"
    # text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
    # de.load_cached_label(cache_path=cache_path, 
    #                     data_style="grounding", 
    #                     text_embed_pt=text_embed_pt)
    # de.print_data_info()

    # de.visual_and_save2(im_index, save_path="./visualized_grounding_example.jpg")


    de=DataEngine()
    cache_path="/root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.updated.cache"
    text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
    de.load_cached_label(cache_path=cache_path, 
                        data_style="grounding", 
                        text_embed_pt=text_embed_pt)
    de.print_data_info()

    de.visual_and_save2(filename="353913.jpg", save_path="./visualized_grounding_example_v2.jpg")


