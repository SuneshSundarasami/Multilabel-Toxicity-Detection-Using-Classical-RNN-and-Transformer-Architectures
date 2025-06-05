import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import argparse

class ToxicCommentGenerator:
    def __init__(self, model_path="./gpt2-toxic-final"):
        """
        Initialize the toxic comment generator
        
        Args:
            model_path (str): Path to the trained model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline
        self.text_generator = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer
        )
        
        print("Model loaded successfully!")
    
    def generate_comment(self, toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0, 
                        max_length=50, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1):
        """
        Generate a toxic comment based on specified toxicity labels
        
        Args:
            toxic, severe_toxic, obscene, threat, insult, identity_hate (int): 0 or 1
            max_length (int): Maximum length of generated text
            do_sample (bool): Whether to use sampling
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p (nucleus) sampling parameter
            num_return_sequences (int): Number of sequences to generate
        
        Returns:
            list: Generated text sequences
        """
        # Create prompt
        labels = f"toxic={toxic}, severe_toxic={severe_toxic}, obscene={obscene}, threat={threat}, insult={insult}, identity_hate={identity_hate}"
        prompt = f"<toxicity> {labels} </toxicity> <comment>"
        
        print(f"Prompt: {prompt}")
        
        # Generate text
        outputs = self.text_generator(
            prompt,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated comments (remove prompt)
        results = []
        for output in outputs:
            generated_text = output['generated_text']
            # Try to extract just the comment part
            if "<comment>" in generated_text:
                comment = generated_text.split("<comment>")[-1].strip()
                results.append(comment)
            else:
                results.append(generated_text)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate toxic comments using trained GPT-2 model')
    parser.add_argument('--model_path', default='./gpt2-toxic-final', help='Path to trained model')
    parser.add_argument('--toxic', type=int, default=0, help='Toxic label (0 or 1)')
    parser.add_argument('--severe_toxic', type=int, default=0, help='Severe toxic label (0 or 1)')
    parser.add_argument('--obscene', type=int, default=0, help='Obscene label (0 or 1)')
    parser.add_argument('--threat', type=int, default=0, help='Threat label (0 or 1)')
    parser.add_argument('--insult', type=int, default=0, help='Insult label (0 or 1)')
    parser.add_argument('--identity_hate', type=int, default=0, help='Identity hate label (0 or 1)')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ToxicCommentGenerator(args.model_path)
    
    # Generate comments
    print(f"\nGenerating {args.num_samples} comments with labels:")
    print(f"toxic={args.toxic}, severe_toxic={args.severe_toxic}, obscene={args.obscene}")
    print(f"threat={args.threat}, insult={args.insult}, identity_hate={args.identity_hate}")
    print("-" * 80)
    
    for i in range(args.num_samples):
        results = generator.generate_comment(
            toxic=args.toxic,
            severe_toxic=args.severe_toxic,
            obscene=args.obscene,
            threat=args.threat,
            insult=args.insult,
            identity_hate=args.identity_hate,
            max_length=args.max_length,
            num_return_sequences=1
        )
        
        print(f"\nGenerated Comment {i+1}:")
        print(f"'{results[0]}'")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE - Enter your own toxicity labels")
    print("="*80)
    
    while True:
        try:
            print("\nEnter toxicity labels (0 or 1 for each):")
            toxic = int(input("Toxic: "))
            severe_toxic = int(input("Severe Toxic: "))
            obscene = int(input("Obscene: "))
            threat = int(input("Threat: "))
            insult = int(input("Insult: "))
            identity_hate = int(input("Identity Hate: "))
            
            results = generator.generate_comment(
                toxic=toxic,
                severe_toxic=severe_toxic,
                obscene=obscene,
                threat=threat,
                insult=insult,
                identity_hate=identity_hate,
                max_length=args.max_length
            )
            
            print(f"\nGenerated Comment:")
            print(f"'{results[0]}'")
            
            continue_input = input("\nGenerate another? (y/n): ").lower()
            if continue_input != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except ValueError:
            print("Please enter valid integers (0 or 1)")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()