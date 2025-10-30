import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from yoloe_data_engine.data_engine import DataEngine

import copy
import numpy as np
from pathlib import Path as _Path
def to_serializable(obj):
    if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


class YoloBox:
    def __init__(self, img_shape: list):
        assert len(img_shape) == 2, "img_sz should be (height,width)"
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.xyxy = None
        self.xywhn = None  # normalized xywh

    def load_from_xywhn_normalized(self, bboxes_xywhn):
        bboxes_xyxy = np.zeros_like(bboxes_xywhn)
        if bboxes_xywhn.shape[0] > 0:
            bboxes_xyxy[:, 0] = (bboxes_xywhn[:, 0] - bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 1] = (bboxes_xywhn[:, 1] - bboxes_xywhn[:, 3] / 2) * self.img_h
            bboxes_xyxy[:, 2] = (bboxes_xywhn[:, 0] + bboxes_xywhn[:, 2] / 2) * self.img_w
            bboxes_xyxy[:, 3] = (bboxes_xywhn[:, 1] + bboxes_xywhn[:, 3] / 2) * self.img_h
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def load_from_xyxy(self, bboxes_xyxy):
        if isinstance(bboxes_xyxy, list):
            bboxes_xyxy = np.array(bboxes_xyxy)
        bboxes_xyxy = np.array(bboxes_xyxy, dtype=np.float32)
        bboxes_xywhn = np.zeros_like(bboxes_xyxy)
        if bboxes_xyxy.shape[0] > 0:
            bboxes_xywhn[:, 0] = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2) / self.img_w
            bboxes_xywhn[:, 1] = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2) / self.img_h
            bboxes_xywhn[:, 2] = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / self.img_w
            bboxes_xywhn[:, 3] = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / self.img_h
        self.xyxy = bboxes_xyxy
        self.xywhn = bboxes_xywhn
        return self

    def iou(self, bbox_xyxy):
        assert self.xyxy is not None, "self.xyxy is None, please load the box first"
        ious = []
        for i in range(self.xyxy.shape[0]):
            box = self.xyxy[i]
            xi1 = max(box[0], bbox_xyxy[0])
            yi1 = max(box[1], bbox_xyxy[1])
            xi2 = min(box[2], bbox_xyxy[2])
            yi2 = min(box[3], bbox_xyxy[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (box[2] - box[0]) * (box[3] - box[1])
            box2_area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            ious.append(iou)
        return np.array(ious)

class Instance:
    def __init__(self, bbox=None, **kwargs):
        self.bbox = bbox
        self.text = None
        self.conf = None
        self.embed = None
        self.vpe = None
        self.segment = None
        self.other_data = {**kwargs}

    def set_segment(self, segment: np.ndarray):
        self.segment = segment
        assert len(segment.shape) == 2 and segment.shape[1] == 2

    def set_embed(self, embed):
        self.embed = embed

    def set_vpe(self, vpe: np.ndarray):
        self.vpe = vpe.squeeze()

    def set_text(self, texts: list, conf: list = None):
        self.text = texts
        self.conf = conf
        assert len(texts) == len(conf)

    def get_top_text_conf(self):
        assert self.text is not None and self.conf is not None
        max_conf_index = np.argmax(self.conf)
        return self.text[max_conf_index], self.conf[max_conf_index]

    def to_dict(self):
        return {
            'bbox': to_serializable(self.bbox),
            'text': to_serializable(self.text),
            'conf': to_serializable(self.conf),
            'embed': to_serializable(self.embed),
            'vp': to_serializable(self.vpe),
            'other_data': to_serializable(self.other_data)
        }

class Sample:
    def __init__(self):
        self.im_file = None
        self.shape = None
        self.instances = []
        self.texts = []
        self.other_data = {}

    def load_from_grounding_label(self, grounding_data: dict):
        self.im_file = grounding_data.get("im_file")
        self.shape = grounding_data.get("shape")
        for text in grounding_data.get("texts"):
            if isinstance(text, list):
                assert len(text) == 1
                self.texts.append(text[0])
            elif isinstance(text, str):
                self.texts.append(text)
            else:
                raise ValueError("text should be str or list of str")
        normalized = grounding_data.get("normalized")
        bbox_format = grounding_data.get("bbox_format")
        self.other_data["bbox_format"] = bbox_format
        self.other_data["normalized"] = normalized
        assert normalized is True
        assert bbox_format == "xywhn"
        for cls, box, segment in zip(grounding_data.get("cls", []), grounding_data.get("bboxes", []), grounding_data.get("segments", [])):
            bbox = YoloBox(self.shape).load_from_xywhn_normalized(np.array([box])).xyxy[0]
            assert bbox.shape == (4,)
            assert segment.shape[0] == 2
            inst = Instance(bbox=box)
            inst.set_segment(segment)
            cls = int(cls)
            assert cls < len(self.texts)
            text = self.texts[cls]
            assert isinstance(text, str)
            inst.set_text([self.texts[cls]], [-1])
            self.instances.append(inst)

    def to_grounding_label(self) -> dict:
        grounding_data = {}
        grounding_data['im_file'] = self.im_file
        grounding_data['shape'] = self.shape
        grounding_data['texts'] = [[text] for text in self.texts]
        bboxes = []
        segments = []
        cls_list = []
        for inst in self.instances:
            bboxes.append(inst.bbox)
            segments.append(inst.segment)
            text, _ = inst.get_top_text_conf()
            cls_index = self.texts.index(text)
            cls_list.append(cls_index)
        grounding_data['bboxes'] = bboxes
        grounding_data['segments'] = segments
        grounding_data['cls'] = cls_list
        grounding_data['normalized'] = True
        grounding_data['bbox_format'] = 'xywhn'
        return grounding_data


    def load_from_yoloe_result(self, yoloe_result):
        self.instances = []
        self.im_file = yoloe_result.path
        self.shape = (yoloe_result.orig_shape[0], yoloe_result.orig_shape[1])
        boxes = yoloe_result.boxes
        names = yoloe_result.names
        for box in boxes:
            bbox_xyxy = box.xyxy.cpu().numpy()
            conf = box.conf.cpu().numpy()
            cls = int(box.cls.cpu().numpy())
            inst = Instance(bbox=bbox_xyxy.tolist())
            inst.set_text([names[cls]], [float(conf)])
            self.instances.append(inst)

    def to_dict(self):
        return {
            'im_file': to_serializable(self.im_file),
            'instances': [inst.to_dict() for inst in self.instances],
            'other_data': to_serializable(self.other_data)
        }
    
    def save_to_json(self, json_path):
        import json
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        # print(f"Saved sample to {json_path}")


class DataEngineAgent:
    def __init__(self, devices=["cuda:0"], buffer_dir="/root/ultra_louis_work/engine_buffer"):
        self.buffer_dir = buffer_dir
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.devices = devices

    def load_model_engine(self):
        # self.model_path = model_path
        self.models = []
        for device in self.devices:
            de = DataEngine(device=device)
            de.load_yoloe()
            self.models.append(de)

    def set_classes(self, texts: list):
        for model in self.models:
            model.set_classes(texts)
        self.texts = texts



    def _batch_model_predict_single_process(self, im_files, engine: DataEngine, **kwargs):
        dst_dir = os.path.join(self.buffer_dir, "model_predict")
        os.makedirs(dst_dir, exist_ok=True)
        conf = kwargs.get("conf", 0.5)
        iou = kwargs.get("iou", 0.4)
        texts = kwargs.get("texts", None)
        if texts is not None:
            engine.set_classes(texts)
        im_names_wo_ext = [os.path.splitext(os.path.basename(im_file))[0] for im_file in im_files]
        dst_files = [os.path.join(dst_dir, f"{name}.json") for name in im_names_wo_ext]
        indices = [i for i in range(len(im_files)) if not os.path.exists(dst_files[i])]
        if len(indices) == 0:
            print("All images have been processed, skip.")
            return
        process_img_files = [im_files[i] for i in indices]
        results = list(engine.model.predict(process_img_files, conf=conf, iou=iou, batch=len(process_img_files), stream=True))
        print(f"Processed {len(process_img_files)} images.")
        for i, sample_index in enumerate(indices):
            sample = Sample()
            result = results[i]
            sample.load_from_yoloe_result(result)
            sample.save_to_json(dst_files[sample_index])
        return
    
    def multi_process_batch_model_predict(self, im_dir, texts=None, conf=0.5, iou=0.4, batch_size=3, max_workers=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        im_files = []
        for file_name in os.listdir(im_dir):
            if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                im_files.append(os.path.join(im_dir, file_name))

        # im_files=im_files[:128]
        print(f"Total images to process: {len(im_files)}")
        batches = [im_files[i:i+batch_size] for i in range(0, len(im_files), batch_size)]
        print(f"Total batches: {len(batches)}, batch size: {batch_size}")
        if max_workers is None:
            max_workers = min(len(self.models), len(batches))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                model = self.models[i % len(self.models)]
                kwargs = {'conf': conf, 'iou': iou, 'texts': texts}
                futures.append(executor.submit(self._batch_model_predict_single_process, batch, model, **kwargs))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Model predict ..."):
                future.result()
        return results

    def _load_grounding_data(self, imid, anns, images, imname_anns, labels, indice, folder_name):
        dst_dir = os.path.join(self.buffer_dir, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, str(imid) + ".json")
        if os.path.exists(dst_file):
            return
        from ultralytics.data.converter import merge_multi_segment
        from ultralytics.data.dataset import segments2boxes
        img = images[f"{imid:d}"]
        h, w, f = img["height"], img["width"], img["file_name"]
        im_file = Path(self.im_dir) / f
        bboxes_xyxy = []
        bboxes = []
        segments = []
        cat2id = {}
        texts = []
        if imname_anns is not None:
            anns_for_img = imname_anns[f]
        else:
            anns_for_img = []
        for ann in anns + anns_for_img:
            if len(bboxes_xyxy) > 0 and YoloBox([int(h), int(w)]).load_from_xyxy(bboxes_xyxy).iou(ann["bbox"]).max() > 0.98:
                continue
            if ann["iscrowd"]:
                continue
            box = np.array(ann["bbox"], dtype=np.float32)
            box[:2] += box[2:] / 2
            box[[0, 2]] /= float(w)
            box[[1, 3]] /= float(h)
            if box[2] <= 0 or box[3] <= 0:
                continue
            caption = ann["caption"]
            cat_name = " ".join([caption[t[0]:t[1]] for t in ann["tokens_positive"]]).lower().strip()
            if not cat_name:
                continue
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append([cat_name])
            cls = cat2id[cat_name]
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
                        s = [j for i in ann["segmentation"] for j in i]
                        s = (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                    s = [cls] + s
                    segments.append(s)
            bboxes_xyxy.append(ann["bbox"])
        lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
        if segments:
            classes = np.array([x[0] for x in segments], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
        lb = np.array(lb, dtype=np.float32)
        label = {
            "im_file": im_file,
            "shape": (h, w),
            "cls": lb[:, 0:1],
            "bboxes": lb[:, 1:],
            "segments": [],
            "normalized": True,
            "bbox_format": "xywh",
            "texts": texts,
        }
        def serializeLabel(label):


            lc = copy.deepcopy(label)
            lc["im_file"] = str(lc.get("im_file", ""))
            lc["shape"] = list(lc.get("shape", []))
            lc["cls"] = to_serializable(lc.get("cls", []))
            try:
                cls_arr = np.array(lc["cls"]).reshape(-1)
                lc["cls"] = [int(x) for x in cls_arr.tolist()]
            except Exception:
                pass
            lc["bboxes"] = to_serializable(lc.get("bboxes", []))
            lc["segments"] = to_serializable(lc.get("segments", []))
            lc["texts"] = to_serializable(lc.get("texts", []))
            lc["normalized"] = bool(lc.get("normalized", True))
            lc["bbox_format"] = str(lc.get("bbox_format", "xywh"))
            return lc
        label_serialized = serializeLabel(label)
        tmp_file = str(dst_file) + ".tmp"
        import json
        with open(tmp_file, "w") as file:
            json.dump(label_serialized, file, indent=4, ensure_ascii=False)
        os.replace(tmp_file, str(dst_file))
        print(f"Saved sample to {dst_file}")

    def multi_thread_load_grounding_data(self, im_dir, json_file, merge_within_one_image, max_workers=8):

        print("Start multi-threaded loading of grounding data...")
        self.im_dir = im_dir
        with open(json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        imid_imname = {f"{im['id']:d}": im["file_name"] for im in annotations["images"]}
        if merge_within_one_image:
            imname_anns = defaultdict(list)
            for ann in annotations["annotations"]:
                imid = ann["image_id"]
                imname = imid_imname[f"{imid:d}"]
                ann["caption"] = images[f"{ann['image_id']:d}"]["caption"]
                imname_anns[imname].append(ann)
            folder_name = "grounding_data_merged"
        else:
            imname_anns = None
            folder_name = "grounding_data"
        imid_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            ann["caption"] = images[f"{ann['image_id']:d}"]["caption"]
            imid_anns[ann["image_id"]].append(ann)
        self.img_path = annotations.get("img_path", "")
        imids = list(imid_anns.keys())
        labels = [None] * len(imids)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._load_grounding_data, imid, imid_anns[imid], images, imname_anns, labels, indice, folder_name) for indice, imid in enumerate(imids)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading grounding data"):
                future.result()
        self.labels = labels

    def _merge_predict(self):
        pass




def read_numpy_and_print(path=None):
    def load_dataset_cache_file(path: Path) -> dict:
        import gc
        gc.disable()
        cache = np.load(str(path), allow_pickle=True).item()
        gc.enable()
        return cache
    path = "/root/ultra_louis_work/engine_buffer/grounding_data/5.cache"
    data = load_dataset_cache_file(path)
    print(data)

if __name__ == "__main__":
    devices = ["cuda:0","cuda:1","cuda:2","cuda:3"]

    # agent = DataEngineAgent(devices=devices, buffer_dir="/root/ultra_louis_work/runs/flickr_engine_buffer")
    # json_file = "/root/ultra_louis_work/datasets/flickr/annotations/final_flickr_separateGT_train_segm.json"
    # im_dir = "../datasets/flickr/full_images/"
    mobileclip_text_embed_pt="/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"


    agent = DataEngineAgent(devices=devices, buffer_dir="/root/ultra_louis_work/runs/mixed_engine_buffer")
    json_file= "../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json"
    im_dir="../datasets/mixed_grounding/gqa/images"
    # mobileclip_text_embed_pt="/root/ultra_louis_work/datasets/mixed_grounding/gqa/text_embeddings_mobileclip_blt.pt"
    mobileclip_text_embed_pt="/root/ultra_louis_work/datasets/flickr/text_embeddings_mobileclip_blt.pt"


    agent.load_model_engine()
    text_list=None
    for index,model in enumerate(agent.models):
        if text_list is None:       
            text_embed_pt = mobileclip_text_embed_pt
            model.set_classes(text_embed_pt=text_embed_pt)
            text_list=model.names
        else:
            model.set_classes(name_list=text_list)

    agent.multi_process_batch_model_predict(im_dir=im_dir, texts=None, conf=0.5, iou=0.4,batch_size=3)


    # agent.multi_thread_load_grounding_data(json_file=json_file, im_dir=im_dir, merge_within_one_image=True, max_workers=1)





