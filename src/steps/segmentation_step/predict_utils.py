import cv2
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation


def make_predict(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-tiny-ade")
    inputs = feature_extractor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-ade")

    output = model(**inputs)
    class_queries_logits = output.class_queries_logits
    masks_queries_logits = output.masks_queries_logits


    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(output, target_sizes=[(480,640)],)[0]

    output = predicted_semantic_map.cpu().detach().numpy()
    return output
