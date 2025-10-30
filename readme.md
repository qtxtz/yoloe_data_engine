


#  pipline of the data engine
###   read the grounding data from json file
    for each sample, per-store the others samples sharing the same image.
    add the 

###   model predict and save the jons files
-     visual to check the json files
-  found that some boxes are overlapped heavily, with different text
- how to deal with these boxes? 

###  merge model prediction to label,
- discard the bbox with higher iou  ( > 0.8, higher iou , no consider the class or text)





-    generate the visual prompt embedding for each instance (bbox)


-    merge bboxes within the same image ( consider the vpe distance  and text similarity ,bbox iou<0.8 )


-    transfer to grounding format cache for training

- 


to do:

write a tools to visual the bbox ious. within the same images 