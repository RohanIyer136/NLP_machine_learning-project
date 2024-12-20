import requests
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm

# Ensure this code is run inside the main guard to avoid multiprocessing errors on Windows
if __name__ == "__main__":
    # Load the pretrained model, tokenizer, and image processor
    # Make sure no local directory named "nlpconnect/vit-gpt2-image-captioning" is in your CWD
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Model in evaluation mode
    model.eval()

    # If needed, you can choose to provide an attention_mask. However, since this model uses a single image at a time
    # and doesn't rely on padded text inputs in inference, you can often ignore the warning.
    # If you still want to provide attention_mask, you need to tokenize with return_tensors="pt" and handle it accordingly.

    # Load your dataset: must contain 'URL' and 'Captions' columns
    # Replace 'your_dataset.csv' with your actual dataset file name
    dataset = load_dataset("csv", data_files="coco_caption_processed.csv")["train"]

    class CaptionEvalDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            url = self.dataset["URL"][idx]
            caption = self.dataset["Captions"][idx]
            return {"URL": url, "Captions": caption}

    eval_dataset = CaptionEvalDataset(dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    predictions = []
    references = []

    # Generate captions for each image in the dataset
    for batch in tqdm(eval_loader, desc="Evaluating"):
        # batch["URL"] is a list of URLs for the current batch
        for url, ref_caption in zip(batch["URL"], batch["Captions"]):
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt").pixel_values

            with torch.no_grad():
                # Generate caption
                # Note: You can provide attention_mask if needed, but for image captioning it's often unnecessary
                # generated_ids = model.generate(pixel_values, attention_mask=...)
                generated_ids = model.generate(pixel_values)
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            predictions.append(generated_text)
            references.append(ref_caption)

    # Compute BLEU score
    bleu = load_metric("bleu")
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )
    print(f"BLEU score: {bleu_score['bleu']}")

    # Compute ROUGE score
    rouge = load_metric("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE results:", rouge_results)
    
    # Compute Meteor score
    meteor = load_metric("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)
    print(f"METEOR score: {meteor_score['meteor']}")

def generate_caption(image_url):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=46,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    # Example inference
sample_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000039785.jpg",
]

for i, url in enumerate(sample_urls, 1):
    caption = generate_caption(url)
    print(f"Sample {i} Caption: {caption}")
    

for pred, ref in zip(predictions[:10], references[:10]):
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    print("-" * 50)