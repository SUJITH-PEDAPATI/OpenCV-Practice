import os
from ultralytics.data.annotator import auto_annotate

# auto_annotate(data = "./Day7 - Auto Annotate Detection Data/dataset/test/images",
#               det_model="./Day7 - Auto Annotate Detection Data/best.pt",
#               sam_model = "sam_b.pt"
#             )

# auto_annotate(data = "./Day7 - Auto Annotate Detection Data/dataset/train/images",
#               det_model="./Day7 - Auto Annotate Detection Data/best.pt",
#               sam_model = "sam_b.pt"
#             )

# auto_annotate(data = "./Day7 - Auto Annotate Detection Data/dataset/val/images",
#               det_model="./Day7 - Auto Annotate Detection Data/best.pt",
#               sam_model = "sam_b.pt"
#             )

import shutil

for dp, dn, _ in os.walk("./Day7 - Auto Annotate Detection Data/dataset"):
    if "images_auto_annotate_labels" in dn:
        src = os.path.join(dp, "images_auto_annotate_labels")
        dst = os.path.join(dp, "labels")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.rename(src, dst)
