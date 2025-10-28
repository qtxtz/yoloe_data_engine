import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from numpy import indices
from pyparsing import annotations
from data_engine import *

from ultralytics.data.dataset import GroundingDataset, DATASET_CACHE_VERSION, save_dataset_cache_file, get_hash,CACHE_SUFFIX,segments2boxes
from ultralytics.data.converter import merge_multi_segment
from collections import defaultdict
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from typing import Any
import json
from ultralytics.models import yolo


class RefineGroundingDataset(GroundingDataset, DataEngine):



    def vpe_text(self, source, visual_prompts ,texts):
        """
        cal the visual prompt embedding for the current image and visual prompts.
        Args:
            source: image source
            visual_prompts: dict, containing "bboxes" and "cls" lists
            texts: list of str, the texts to be matched
        Returns:
            matched texts for each box: tensor, (N,)
        """


        yoloe_model= self.model
        predictor=yolo.yoloe.YOLOEVPDetectPredictor
        if type(yoloe_model.predictor) is not predictor:
            yoloe_model.predictor = predictor(
                overrides={
                    "task": yoloe_model.model.task,
                    "mode": "predict",
                    "save": False,
                    "batch": 1,
                },
                _callbacks=yoloe_model.callbacks,
            )
        # self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
            

                # get the vpe from current image and visual prompts
        prompts={"bboxes": visual_prompts["bboxes"],
                    "cls":list( range( len(visual_prompts["cls"])))}
        num_cls= len(set(prompts["cls"]))
        yoloe_model.model.model[-1].nc = num_cls
        yoloe_model.model.model[-1].no = num_cls + yoloe_model.model.model[-1].reg_max * 4
        yoloe_model.model.names = [f"object{i}" for i in range(num_cls)]
        yoloe_model.predictor.set_prompts(prompts.copy())
        yoloe_model.predictor.setup_model(model=yoloe_model.model)
        vpe = yoloe_model.predictor.get_vpe(source).squeeze(0)

        tpe= yoloe_model.get_text_pe(texts).squeeze(0)

        # normalize
        vpe= torch.nn.functional.normalize(vpe,dim=-1,p=2)
        tpe= torch.nn.functional.normalize(tpe,dim=-1,p=2)
        # cal the similarity and return the text for each box
        similarities = (vpe @ tpe.T).softmax(dim=-1)  # (N, M)
        matched_indices = similarities.argmax(dim=-1)  # (N,)
        matched_texts = [texts[i] for i in matched_indices.tolist()]
        return matched_texts



    

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)


        # images = {f"{im['id']:d}": im for im in annotations["images"]}

        # Map image IDs to file names
        imid_imname = {f"{im['id']:d}": im["file_name"] for im in annotations["images"]}

        # map image names to annotations
        imname_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            imid = ann["image_id"]
            imname = imid_imname[f"{imid:d}"]
            imname_anns[imname].append(ann)
        



        # # map sample id to the annotations
        # img_ids= [im["id"] for im in annotations["images"]]
        # imid_anns= defaultdict(list)
        # for id in img_ids:
        #     imid_anns[id]=imname_anns[imid_imname[f"{id:d}"]]

        images = {f"{x['id']:d}": x for x in annotations["images"]}
        imid_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            imid_anns[ann["image_id"]].append(ann)


        if not hasattr(self, 'model') or self.model is None:
            self.load_yoloe()


        for img_id, anns in TQDM(imid_anns.items(), desc=f"Reading annotations {self.json_file}"):

            # if img_id > 16: break  # for testing
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes_xyxy=[]
            bboxes = []
            segments = []
            cat2id = {}
            texts = []


            anns_for_img=imname_anns[f]

            for ann in anns + anns_for_img:

                if len(bboxes_xyxy) > 0 and YoloBox([int(h),int(w)]).load_from_xyxy(bboxes_xyxy).iou(ann["bbox"]).max()>0.98:
                    # print("skip duplicate box")
                    continue
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
                bboxes_xyxy.append(ann["bbox"]) # add xyxy box for iou calculation


            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            label=  {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }

            #


            x["labels"].append(label)


        
        #######  append boxes 

        batch_size=32

        for start in tqdm(range(0,len(x["labels"]),batch_size)):
            batch_indices=list(range(start,min(start+batch_size,len(x["labels"]))))
            batch_texts=[]
            for indice in batch_indices:
                label_texts = x["labels"][indice].get("texts", [])
                if isinstance(label_texts, list):
                    for entry in label_texts:
                        if isinstance(entry, (list, tuple)):
                            batch_texts.extend(str(t) for t in entry)
                        else:
                            batch_texts.append(str(entry))

            if batch_texts:
                unique_texts = list(dict.fromkeys(batch_texts))
                self.set_classes(name_list=unique_texts)
            else:
                self.set_classes(name_list=None)


        self.data_style="grounding"

        indices = list(range(len(x["labels"])))
        results=self.yoloe_predict_batch([ x["labels"][i] for i in indices ], conf=0.01)
        assert len(results)==len(indices), "Mismatch between results and indices length"
        for indice,res in zip(indices,results):
            x["labels"][indice]= self._update_grounding_label(x["labels"][indice],res,iou=0.5,replace=False)
        


        #####  refine the bbox texts
        imname_image = {im["file_name"]: im for im in annotations["images"]}
        for indice,label in tqdm(enumerate(x["labels"]), desc="Refining texts for grounding data"):
            bboxes_xyxy= YoloBox((int(label["shape"][0]),int(label["shape"][1]))).load_from_xywhn_normalized(label["bboxes"]).xyxy
            visual={"bboxes": bboxes_xyxy,
                    "cls": list(range(bboxes_xyxy.shape[0]))}
            texts= []
            for text_list in label["texts"]:
                texts.extend(text_list)
            print("original texts for image ",  ":", texts)
            caption= imname_image[label["im_file"].name]["caption"]
            caption_texts= caption.split()
            texts.extend(caption_texts)
            print("caption_texts for image ",  ":", caption_texts)
            texts= list(set(texts))
            matched_texts= self.vpe_text(source= label["im_file"], visual_prompts= visual, texts= texts)
            matches_texts_set= list(set(matched_texts))
            label['texts']= [[text] for text in matches_texts_set]
            # take cls as the index in the matched texts set
            label["cls"]= [   matches_texts_set.index(text) for text in matched_texts ]
            print(label['cls'])
            print(matched_texts)
            print(label['texts'])
            x["labels"][indice]= label


        x["hash"] = get_hash(self.json_file)


        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x






# """
# refine the text for grounding data by running grounding prediction and updating the texts.
# how to set classes:
# 1. collect all texts in the current batch.
# 2. do such refinement for each image in the batch: 


# # read from the json file as 

# ["two people"] -> ["two", "people","two people"]
# ["what"]. -> update according to the grounding prediction results.


# """




# def load_src_json(self, json_path):

#     """
#     read the original json file for grounding dataset. 

#     """
#     import json
#     with open(json_path, 'r') as f:
#         data = json.load(f) #. dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

#     print(data.keys())
#     print(data["info"])
#     print(data["categories"])

#     print("-"*40)
#     imgname2imgid={}
#     for img in data["images"]:
#         imgname = img["file_name"]
#         if imgname not in imgname2imgid.keys():
#             imgname2imgid[imgname]=[]
#         imgname2imgid[imgname].append(img["id"])
#     print("number of images:", len(imgname2imgid.keys()))


#     print("number of samples:", len(data["images"]))


#     print("number of annotations:", len(data["annotations"]))
#     print("-"*40)
#     print("example image entry:", data["images"][0])
#     print("exmple category entry:")

#     for i in range(4):
#         print("-"*40)
#         print( data["annotations"][i]) 


#     return 
#     json_data = {
#         "file_names": [],
#         "images": {},
#         "annotations": {} 
#     }
#     for x in data["images"]:
#         file_name = x["file_name"]
#         json_data["file_names"].append(file_name)
#         if file_name not in json_data["images"]:
#             json_data["images"][file_name] = []
#         json_data["images"][file_name].append(x)
#         print

#     self.json_data = json_data
    
# def get_captions_texts(self,file_name):
#     """
#      get all captions and split them into texts
#     """
#     captions = []
#     caption_texts = []
#     if file_name in self.json_data["images"]:
#         for img_entry in self.json_data["images"][file_name]:
#             caption= img_entry.get("caption", "")
#             captions.append(caption)
#             texts= caption.split()
#             caption_texts.extend(texts)
#     print(captions)
#     print(caption_texts)
#     return captions, caption_texts



# de=DataEngine()

# load_src_json(de, "/root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.json")

# cache_path="/root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.merged.cache"
# text_embed_pt="/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"
# # de.load_cached_label(cache_path=cache_path, 
# #                     data_style="grounding", 
# #                     text_embed_pt=text_embed_pt)

# # de.load_yoloe()
# file_name=de.json_data["file_names"][0]

# get_captions_texts(de, file_name)

# # def predict_and_update_text(self,indice):



from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPTrainer





DATA_DIR="../datasets/"

Objects365v1="../datasets/Objects365v1.yaml"



data= RefineGroundingDataset(
        img_path=DATA_DIR+"flickr/full_images/",
        json_file=DATA_DIR+"flickr/annotations/final_flickr_separateGT_train_segm.json",
    )






# data= RefineGroundingDataset(
#         img_path=DATA_DIR+"mixed_grounding/gqa/images",
#         json_file=DATA_DIR+"mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
#     )