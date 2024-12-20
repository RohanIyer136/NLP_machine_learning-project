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
    
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset
    dataset = load_dataset("csv", data_files="coco_caption_processed.csv")
    train_test_split= dataset["train"].train_test_split(test_size=0.2, seed=42)
    # Access train and test sets
    train_data = train_test_split["train"]
    eval_data = train_test_split["test"]
    
    class CaptionDataset(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets

        def __len__(self):
            return len(self.datasets)

        def __getitem__(self, idx):
            # Get the image URL for the current data sample
            image_url = self.datasets['URL'][idx]
            
            # Load the image
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")  # Ensure all images are RGB
            
            # Process the image for the Vision Transformer (ViT) model
            image_features = image_processor(image, return_tensors="pt").pixel_values

            # Tokenize the caption
            labels = tokenizer(
                self.datasets["Captions"][idx],
                return_tensors="pt",
                max_length=46,
                padding="max_length",
                truncation=True
            ).input_ids
            
            return {'pixel_values': image_features.squeeze(0), 'labels': labels.squeeze(0)}

    #print(datasets)
    datasets_train = CaptionDataset(train_data)
    eval_dataset = CaptionDataset(eval_data) #, tokenizer, image_processor)


    ### TRAIN ###

    from transformers import Trainer, TrainingArguments
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Set GPU if available, else CPU
    # model = model.to(device)  # Move model to appropriate device (if not done already)

    model.train()  # Sets the model into training mode

    training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device during training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,  # Save only 2 checkpoints to save space
    evaluation_strategy="steps",
    fp16=True,
    )

    trainer = Trainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments
    train_dataset=datasets_train,        # Training dataset
    eval_dataset=eval_dataset            # If you have a validation dataset, use it here
    )
    trainer.train()
    # Model in evaluation mode
    model.eval()

    
    # eval_dataset = CaptionDataset(dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # Evaluation: Use original dataset instead of processed DataLoader
    predictions = []
    references = []

    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        # Access original data from eval_dataset
        data = eval_dataset.datasets[idx]  # Get raw data for this index
        url = data["URL"]
        ref_caption = data["Captions"]

        # Load and process image
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
        except Exception as e:
            print(f"Error loading image {url}: {e}")
            continue

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Store prediction and reference
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