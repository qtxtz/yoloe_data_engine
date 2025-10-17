import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


import numpy as np
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont

from ultralytics.data.utils import load_dataset_cache_file
from ultralytics.engine.results import Results, Boxes

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_names_from_yaml_config(yaml_config):
    import yaml
    if not os.path.exists(yaml_config):
        raise FileNotFoundError(f"YAML config file not found: {yaml_config}")
    with open(yaml_config, 'r') as f:
        data_dict = yaml.safe_load(f)
        names = data_dict['names']
    return names


class YoloBox(object):

    def __init__(self,img_shape:list):
        assert len(img_shape)==2, "img_sz should be (height,width)"
        self.img_h=img_shape[0]
        self.img_w=img_shape[1]
        self.xyxy=None
        self.xywhn=None # normalized xywh

    def load_from_xywhn_normalized(self,bboxes_xywhn):
        # xywhn: [N,4]  x_center,y_center,w,h (normalized)
        bboxes_xyxy = np.zeros_like(bboxes_xywhn)
        if bboxes_xywhn.shape[0] > 0:
            bboxes_xyxy[:, 0] = (bboxes_xywhn[:, 0] - bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 1] = (bboxes_xywhn[:, 1] - bboxes_xywhn[:, 3] / 2) * self.img_h
            bboxes_xyxy[:, 2] = (bboxes_xywhn[:, 0] + bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 3] = (bboxes_xywhn[:, 1] + bboxes_xywhn[:, 3] / 2) * self.img_h

        self.xyxy=bboxes_xyxy

        self.xywhn=bboxes_xywhn
        return self
    def load_from_xyxy(self,bboxes_xyxy):
        # xyxy: [N,4]  x0,y0,x1,y1
        bboxes_xywhn = np.zeros_like(bboxes_xyxy)
        if bboxes_xyxy.shape[0] > 0:
            bboxes_xywhn[:, 0] = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2) / self.img_w
            bboxes_xywhn[:, 1] = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2) / self.img_h
            bboxes_xywhn[:, 2] = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / self.img_w
            bboxes_xywhn[:, 3] = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / self.img_h

        self.xyxy=bboxes_xyxy

        self.xywhn=bboxes_xywhn
        return self


import torch

class DataEngine:
    
    def __init__(self,device="cuda"):
        self.device=device



    def load_yoloe(self):
        from ultralytics import YOLOE
        model_path="/root/ultra_louis_work/ultralytics/yoloe-11l-seg.pt"
        self.model=YOLOE("yoloe-11l.yaml").load(model_path)
        print("load model from:", model_path)

    def set_classes(self,yaml_config=None,name_list=None, text_embed_pt=None):

        # only one of yaml_config and name_list should be provided
        assert (yaml_config is None) or (name_list is None), "Only one of yaml_config and name_list should be provided"
        if yaml_config is not None:

            assert name_list is None, "If yaml_config is provided, name_list should be None"
            name_list = get_names_from_yaml_config(yaml_config)
            name_list=list(name_list.values())
            print("Load names from yaml:", yaml_config)

        if text_embed_pt is not None:
            assert os.path.exists(text_embed_pt), f"Text embed pt file not found: {text_embed_pt}"
            txt_map= torch.load(text_embed_pt, map_location=self.device)
            name_list=list(txt_map.keys())
            print("Load text embed from:", text_embed_pt)


        assert name_list is None or isinstance(name_list,list), "name_list should be a list of strings or None"

        if name_list is not None :
            self.model.set_classes(name_list, self.model.get_text_pe(name_list))
            print(f"Set {len(name_list)} classes")
        else:
            print("No classes set")



    def yoloe_predict(self,indice,conf=0.05,save_path=None):
        
        img_file=self.labels[indice]['im_file']
        result=self.model.predict(img_file,conf=conf)

        if save_path is not None:
            result[0].save(save_path)
            print("save to:", save_path)
        return result
    

    def __len__(self):
        return len(self.labels)




    def set_img_folder(self,img_source):
        self.img_source=img_source

    def load_cached_label(self,cache_path, data_style="grounding",yaml_config=None, text_embed_pt=None ):
        
        self.cache_path=cache_path

        cache=load_dataset_cache_file(Path(cache_path))
        self.cache=cache
        self.labels=cache["labels"]
        print(len(self.labels))

        assert data_style in ["grounding","detection"]
        self.data_style=data_style
        
        if data_style=="detection":
            assert yaml_config is not None, "yaml_config must be provided for detection data_style"
            if not os.path.exists(yaml_config):
                raise FileNotFoundError(f"YAML config file not found: {yaml_config}")
            self.yaml_config=yaml_config

                 # read names from the  yaml file
            import yaml
            with open(yaml_config, 'r') as f:
                data_dict = yaml.safe_load(f)
                self.names = data_dict['names']

        elif data_style=="grounding":
            assert text_embed_pt is not None, "text_embed_pt must be provided for grounding data_style"
            self.text_embed_pt=text_embed_pt
            if not os.path.exists(text_embed_pt):
                raise FileNotFoundError(f"Text embed pt file not found: {text_embed_pt}")
            else:
                print("Load text embed from:", text_embed_pt)
            txt_map = torch.load(cache_path, map_location=self.device,weights_only=False)
            self.names=list(txt_map.keys())
            
    def save_cached_label(self,save_path=None):
        if save_path is None:
            save_path=self.cache_path
        from copy import deepcopy
        copy_cache=deepcopy(self.cache)
        copy_cache["labels"]=self.labels
        with open(save_path, "wb") as f:
            np.save(f, copy_cache)

    def print_one_label(self,indice):
        
        label=self.labels[indice]
        # print(self.labels[indice])
        print(label.keys())
        print(label['im_file'])
        for key,val in label.items():
            print(f"{key}: {type(val)}")
            if isinstance(val,list):
                print(f"  Length: {len(val)}")
                # if len(val)>0:
                #     print(f"  First 3 elements: {val[:3]}")

            elif isinstance(val,np.ndarray):
                print(f"  Shape: {val.shape}")
                print(f"  Dtype: {val.dtype}")
                print(f"  First 5 elements: {val.flatten()[:5]}")
            
            elif isinstance(val,dict):
                print(f"  Dict with keys: {list(val.keys())}")

            else:
                print(f"  Value: {val}")


    def detection_predict_and_update_labels(self,indice,iou=0.3,replace=True):
        # im_file=self.labels[indice]['im_file']
        result=self.yoloe_predict(indice=indice,conf=0.1)

        assert self.data_style == "detection", "detection_predict_and_update_labels only works for detection data_style"
        boxes=result[0].boxes
        bboxes_xyxy=boxes.xyxy.cpu().numpy()
        yolo_box=YoloBox(img_shape=result[0].orig_img.shape[:2]).load_from_xyxy(bboxes_xyxy)

        bboxes_xywhn=yolo_box.xywhn
        cls=boxes.cls.cpu().numpy()
        assert bboxes_xywhn.shape[0]==cls.shape[0], "Mismatch between number of boxes and classes"
        if replace:
            self.labels[indice]['bboxes']=bboxes_xywhn
            self.labels[indice]['cls']=cls
            print(f"Replace with {bboxes_xywhn.shape[0]} boxes")
        else:
            # compare the new boxes with existing boxes, and append only the new ones
            # if iou less than threshold
            keep_indices=[]
            for i in range(bboxes_xywhn.shape[0]):
                bbox=bboxes_xywhn[i]
                c=cls[i]
                max_iou=0
                for j in range(self.labels[indice]['bboxes'].shape[0]):
                    exist_bbox=self.labels[indice]['bboxes'][j]
                    # compute iou
                    box1=YoloBox(img_shape=result[0].orig_img.shape[:2]).load_from_xywhn_normalized(bbox).xyxy[0]
                    box2=YoloBox(img_shape=result[0].orig_img.shape[:2]).load_from_xywhn_normalized(exist_bbox).xyxy[0]
                    # box: x0,y0,x1,y1
                    xi1 = max(box1[0], box2[0])
                    yi1 = max(box1[1], box2[1])
                    xi2 = min(box1[2], box2[2])
                    yi2 = min(box1[3], box2[3])
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou>max_iou:
                        max_iou=iou
                if max_iou<iou:
                    keep_indices.append(i)
            print(f"Append {len(keep_indices)} new boxes out of {bboxes_xywhn.shape[0]}")   

            for i in keep_indices:
                bbox=bboxes_xywhn[i]
                c=cls[i]
                self.labels[indice]['bboxes']=np.vstack([self.labels[indice]['bboxes'],bbox])
                self.labels[indice]['cls']=np.hstack([self.labels[indice]['cls'],c])

    def grounding_predict_and_update_labels(self,indice,iou=0.05,replace=True):


        im_file=self.labels[indice]['im_file']


        result=self.yoloe_predict(im_file,conf=0.1)
        assert self.data_style == "grounding", "grounding_predict_and_update_labels only works for grounding data_style"
        boxes=result[0].boxes
        bboxes_xyxy=boxes.xyxy.cpu().numpy()
        yolo_box=YoloBox(img_shape=result[0].orig_img.shape[:2]).load_from_xyxy(bboxes_xyxy)
        bboxes_xywhn=yolo_box.xywhn
        cls=boxes.cls.cpu().numpy()      

        assert bboxes_xywhn.shape[0]==cls.shape[0], "Mismatch between number of boxes and classes"

        if replace:
            self.labels[indice]['bboxes']=bboxes_xywhn
            self.labels[indice]['cls']=cls
            print(f"Replace with {bboxes_xywhn.shape[0]} boxes")  





    def label_append_instance(self,indice,bboxes,cls,texts=None):
        
        assert len(bboxes)==len(cls), "Length of bboxes and cls must be the same"
        



    def visual_and_save2(self, indice,
                          save_path="./visualize2.jpg"):
        """Visualizes a label using ultralytics.engine.results.Results and saves it."""
        
        assert self.data_style in ["grounding","detection"]
        
        if self.data_style=="detection":
            assert self.yaml_config is not None, "yaml_config must be provided for detection data_style"


        

        label = self.labels[indice]
        print("label keys:", label.keys())
        im_file = label['im_file']

        if hasattr(self, 'img_source'):
            im_file = os.path.join(self.img_source, im_file)

        orig_img = np.array(Image.open(im_file))
        img_h, img_w = orig_img.shape[:2]
        
        bboxes_xywhn = label['bboxes']
        cls = label['cls']
        if self.data_style=="detection":
            # transfer cls to texts
            texts = [str(int(c.item())) for c in cls]
            names=self.names
        else:
            texts = label['texts']
            assert isinstance(texts, list), "Expected texts to be a list"
            names = {i: (t[0] if isinstance(t, list) else t) for i, t in enumerate(texts)}
        # Convert from normalized [x_center, y_center, w, h] to [x0, y0, x1, y1]
        bboxes_xyxy = YoloBox(img_shape=(img_h, img_w)).load_from_xywhn_normalized(bboxes_xywhn).xyxy

        plot_cls = cls

        # Create Boxes object
        # We need to add confidence and class to each bbox.
        # Format for Boxes is [x, y, w, h, conf, cls]
        # Using dummy confidence of 1.0
        conf = np.ones((bboxes_xywhn.shape[0], 1))
        
        # Ensure plot_cls is a column vector (N, 1)
        if plot_cls.ndim == 1:
            plot_cls = plot_cls[:, np.newaxis]

        boxes_data = np.hstack([bboxes_xyxy, conf, plot_cls])
        
        # Create Results object
        result = Results(
            orig_img=orig_img,
            path=im_file,
            names=names,
            boxes=boxes_data
        )
        
        if result.boxes:
            result.boxes.is_track = False # Set to false to avoid printing track_ids

        result.save(save_path)
        # # Plot the results

        # im_array = result.plot(conf=False) # conf=False to not show confidence scores
        
        # # Save the visualized image
        # Image.fromarray(im_array[..., ::-1]).save(save_path)  # BGR to RGB for PIL
        print(f"Saved visualization to {save_path}")


de=DataEngine()


yaml_config="/root/ultra_louis_work/datasets/Objects365v1.yaml"

cache_path="/root/ultra_louis_work/datasets/Objects365v1/labels/train.cache"
de.load_cached_label(cache_path=cache_path, data_style="detection", yaml_config=yaml_config)
de.load_yoloe()

de.set_classes(yaml_config=yaml_config)

# de=DataEngine()
# cache_path="/root/ultra_louis_work/datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.merged.cache"
# text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
# de.load_cached_label(cache_path=cache_path, 
#                      data_style="grounding", 
#                      text_embed_pt=text_embed_pt)
# de.load_yoloe()

# de.set_classes(yaml_config=None,  text_embed_pt=text_embed_pt)

from tqdm import tqdm

for indice in tqdm(range(len(de))):

    # de.visual_and_save2(indice,
    #                     save_path=f"./visualize_before_{indice}.jpg")
    de.detection_predict_and_update_labels(indice,iou=0.1)
de.save_cached_label(save_path=cache_path.replace(".cache", "_updated.cache"))



# de.load_yoloe(yaml_config="/root/ultra_louis_work/datasets/Objects365v1.yaml")



# de.visual_and_save2(1000,data_style="detection",
#                     yaml_config="/root/ultra_louis_work/datasets/Objects365v1.yaml")

# de.yoloe_predict(1000,conf=0.25,save_path="./yoloe_pred.jpg")


# de.print_one_label(1000)