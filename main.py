from src.features.image import ImageFeature

image_path = "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/cats/1.jpg"

# image = ImageFeature(image_path=image_path, name="image")
# image.generate_cls_masks("cat")
# image.save_xor_plot(plot=False)
# cats = image.cls_masks["cat"]

cls = "cat"
image = ImageFeature(image_path=image_path, name=f"{cls} image")
image.generate_cls_masks(cls)
cats = image.cls_masks[cls]
mask = cats[0]

mask_image_manager = mask.image_manager
mask_image_manager.generate_object_mask(save=True, padding=0)
# mask_image_manager.generate_transparent_mask(save=True, padding=10)
image.save_xor_plot(save=True)
