import torch  
from transformers import CLIPModel, CLIPProcessor 
from PIL import Image  
import requests  
from io import BytesIO  



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_image_from_url(url):
    response = requests.get(url)  
    return Image.open(BytesIO(response.content))  


def load_image_from_file(file_path):
    return Image.open(file_path)  


def text_to_image_similarity(model, processor, text_queries, images):
    text_inputs = processor(text=text_queries, return_tensors='pt', padding=True)
    with torch.no_grad():  
        text_features = model.get_text_features(**text_inputs)  
        text_features /= text_features.norm(dim=-1, keepdim=True)  

    
    image_inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():  
        image_features = model.get_image_features(**image_inputs)  
        image_features /= image_features.norm(dim=-1, keepdim=True)  

    ### text_features -> shape(4, 1028) ; image_features -> shape(2, 1028)
    print(f'Text shape after processing: {text_features.shape}, Image shape after processing: {image_features.shape}')
    
    similarity = (text_features @ image_features.T) ### shape(4, 2)
    return similarity.cpu().numpy()  


def main():
    device = 'cpu'
    print("\nExample 1: Comparing text prompts to online images")
    image_urls = [
        "https://cdn.shopify.com/s/files/1/0086/0795/7054/files/Golden-Retriever.jpg?v=1645179525",  
        "https://miro.medium.com/v2/resize:fit:1400/1*tMKkGydXuiOBOb15srANvg@2x.jpeg",
        "https://m.media-amazon.com/images/I/61nzzgqh-mS.jpg"
    ]

    
    try:
        images = [load_image_from_url(url) for url in image_urls]  
        print(f"Successfully loaded {len(images)} images")
    except Exception as e:
        print(f"Error loading images: {e}")  
        print("Falling back to local images if available...")
        
        return

    
    text_queries = ["a dog, golden retriever", "a car", "a sunset on the beach", "a person"]

    
    similarities = text_to_image_similarity(model, processor, text_queries, images)

    
    print("\nSimilarity Results (%):")
    for i, text in enumerate(text_queries):
        print(f"\nText: '{text}'")
        for j, url in enumerate(image_urls):
            print(f"  Image {j + 1}: {similarities[i][j] * 100:.2f}%")  

    
    print("\nExample 2: Zero-shot image classification")


    labels = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a photo of a sunset on the beach", "funny cat in glasses"]


    # image_input = processor(images=images[0], return_tensors="pt")
    image_input = processor(images=images[1], return_tensors="pt")

    text_inputs = processor(text=labels, return_tensors='pt', padding=True)


    with torch.no_grad():
        image_features = model.get_image_features(**image_input)
        text_features = model.get_text_features(**text_inputs)


        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)


        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits_per_image.cpu().numpy()[0]


    print("\nClassification Results:")
    for i, label in enumerate(labels):
        print(f"{label}: {probs[i] * 100:.2f}%")


if __name__ == "__main__":
    main()  