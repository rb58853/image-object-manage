from src.features.image import ImageFeature

image_path = "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/cats/1.jpg"

image = ImageFeature(image_path=image_path, name="image")
image.generate_cls_masks("cat")
image.save_xor_plot(plot=False)
cats = image.cls_masks["cat"]
