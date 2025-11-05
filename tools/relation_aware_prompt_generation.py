from openai import OpenAI
import json
import jsonlines
import time
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def init_openai_client(api_key: str) -> OpenAI:
    """
    Initialize OpenAI client with the provided API key.
    
    Args:
        api_key: OpenAI API key for authentication.
    
    Returns:
        Initialized OpenAI client instance.
    """
    return OpenAI(api_key=api_key)


def load_entities_and_predicates(dataset: str) -> tuple[list[str], list[str]]:
    """
    Load predefined super entities and predicates based on the target dataset.
    
    Args:
        dataset: Name of the dataset (supports "vg" or "oiv6").
    
    Returns:
        Tuple containing two lists: [super_entities, predicates]
    """
    if dataset == "vg":
        super_entities = [
            'male', 'female', 'children', 'pets', 'wild animal', 'ground transport',
            'water transport', 'air transport', 'sports equipment', 'seating furniture',
            'decorative item', 'table', 'upper body clothing', 'lower body clothing',
            'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food', 'beverage',
            'utensils', 'container', 'textile', 'landscape', 'urban feature', 'plant',
            'structure', 'household item', 'head part', 'limb and appendage'
        ]
        predicates = [
            "above", "across", "against", "along", "and", "at", "attached to", "behind",
            "belonging to", "between", "carrying", "covered in", "covering", "eating",
            "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
            "in", "in front of", "laying on", "looking at", "lying on", "made of",
            "mounted on", "near", "of", "on", "on back of", "over", "painted on",
            "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on",
            "to", "under", "using", "walking in", "walking on", "watching", "wearing",
            "wears", "with"
        ]
    elif dataset == "oiv6":
        super_entities = [
            'male', 'female', 'children', 'head feature', 'limb feature', 'torso feature',
            'accessorie', 'mammal', 'bird', 'reptile', 'insect', 'marine animal', 'bike',
            'ground vehicle', 'watercraft', 'aircraft', 'vehicle part item', 'ball-relatedsport item',
            'water sport item', 'winter sportitem', 'seating furniture', 'table furniture',
            'storage furniture', 'bedding', 'upper bodyclothing', 'lower body clothing',
            'footwear', 'fruit', 'vegetable', 'prepared food', 'beverage', 'appliance',
            'utensil', 'decorative item', 'textile', 'hand tool', 'power tool', 'kitchentool',
            'personal electronic', 'home electronic', 'office electronic', 'land vehicle',
            'watervehicle', 'air vehicle', 'string instrument', 'wind instrument',
            'percussion instrument', 'firearm', 'container', 'toy', 'stationery',
            'landscape', 'urban feature'
        ]
        predicates = [
            "at", "holds", "wears", "surf", "hang", "drink", "holding hands", "on", "ride",
            "dance", "skateboard", "catch", "highfive", "inside of", "eat", "cut", "contain",
            "handshake", "kiss", "talk on phone", "interacts with", "under", "hug", "throw",
            "hits", "snowboard", "kick", "ski", "plays", "read"
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'vg' or 'oiv6' instead.")
    
    return super_entities, predicates


def build_prompt(subject: str, object_: str, predicate: str) -> str:
    """
    Build the prompt template for OpenAI API, including examples and target triplet.
    
    Args:
        subject: Subject entity in the triplet.
        object_: Object entity in the triplet (avoid conflict with Python keyword 'object').
        predicate: Predicate in the triplet.
    
    Returns:
        Formatted prompt string.
    """
    prompt_template = """
Describe [subject] [predicate] [object] which parts of subject and object function in this relationship. Please list these parts, and then analyze and describe the visual relationship between these parts. The generated description should be concise and clear. Here are two examples for you to learn:

Example A: “[human] [holding] [wild animal]”:   
    Subject Part : [hand, arm, legs, ...]
    Object Part : [animal limbs, animal body, ...]
    Region Rescriptions: [“human hand(s) securely gripping the animal”, “human arm(s) embracing or supporting the animal”, “animal positioned close to or physically touching the human’s torso”, “animal appears stable and not struggling”, “direct gaze or interaction between the human and the animal suggesting control or care”, “human fingers intertwined or wrapped around the animal’s body or limbs”, “animal’s posture conveys being held, often with limbs tucked or supported”, “proximity of the human face to the animal, especially when holding smaller animals”, “human holding the animal with hands”, “human’s hands or arms in contact with the animal”, “animal is held in the human’s arms”]
Example B: “[human] [sitting on] [seating furniture]”:
    Subject Part : [buttocks, thighs, legs, back, arms]
    Object Part : [seat, backrest, armrests]
    Region Rescriptions : [“Human’s buttocks are making contact with the seat of the furniture.”, “Human’s thighs rest on the seat, with legs positioned either bent or extended.”, “Human’s back is supported by the backrest of the furniture.”, “Human’s arms may be resting on or near the armrests of the furniture, if present.”, “The furniture’s seat aligns with the human’s buttocks and thighs, indicating proper seating support.”, “The human’s posture is influenced by the backrest, which can be either upright or reclining.”, “The armrests, if present, support the human’s arms, enhancing comfort and stability.”, “The arrangement of the human’s legs and feet suggests their interaction with the seat and alignment with the furniture.”]

Now, the subject: [{}], object: [{}], predicate: [{}], giving the response.
    """
    return prompt_template.format(subject, object_, predicate).strip()


def process_chunk(
    chunk: list[tuple[str, str, str]],
    worker_id: int,
    client: OpenAI,
    output_prefix: str,
    model_name: str,
    temperature: float,
    retry_delay: int
) -> dict[str, str]:
    """
    Process a chunk of triplets (subject-obj-pred) to generate descriptions via OpenAI API.
    Results are saved to jsonl files in real-time.
    
    Args:
        chunk: List of triplets, each in (subject, object, predicate) format.
        worker_id: Unique ID for the worker (used in output filename).
        client: Initialized OpenAI client.
        output_prefix: Prefix path for output jsonl files.
        model_name: Name of the OpenAI model to use (e.g., "gpt-4o-mini").
        temperature: Temperature parameter for API (controls randomness).
        retry_delay: Delay (in seconds) before retrying after an error.
    
    Returns:
        Dictionary mapping triplet strings to generated descriptions.
    """
    chunk_results = {}
    output_path = f"{output_prefix}worker_{worker_id}.jsonl"
    
    for sub, obj, pred in chunk:
        triplet = f"{sub}_{pred}_{obj}"
        prompt = build_prompt(sub, obj, pred)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            description = response.choices[0].message.content
            chunk_results[triplet] = description
            
            # Save to jsonl (append mode)
            with jsonlines.open(output_path, mode='a') as writer:
                writer.write({
                    "triplet": triplet,
                    "rel_text": description,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        except Exception as e:
            # Log error and retry after delay
            error_msg = f"[{datetime.now()}] Worker {worker_id} failed to process {triplet}: {str(e)}"
            print(error_msg)
            time.sleep(retry_delay)
    
    return chunk_results


def split_tasks(tasks: list[tuple[str, str, str]], num_workers: int) -> list[list[tuple[str, str, str]]]:
    """
    Split all tasks into equal chunks for parallel processing.
    
    Args:
        tasks: List of all triplet tasks.
        num_workers: Number of workers (determines number of chunks).
    
    Returns:
        List of chunks, each being a subset of the original tasks.
    """
    return [tasks[i::num_workers] for i in range(num_workers)]


def main(args):
    # 1. Initialize dependencies
    client = init_openai_client(args.openai_api_key)
    super_entities, predicates = load_entities_and_predicates(args.dataset)
    
    # 2. Generate all tasks (subject-object-predicate triplets)
    print(f"Total entities: {len(super_entities)}, Total predicates: {len(predicates)}")
    all_tasks = [
        (sub, obj, pred) 
        for sub in super_entities 
        for obj in super_entities 
        for pred in predicates
    ]
    print(f"Total triplet tasks: {len(all_tasks)}")
    
    # 3. Split tasks into chunks
    task_chunks = split_tasks(all_tasks, args.max_workers)
    print(f"Split into {len(task_chunks)} chunks (1 per worker)")
    
    # 4. Parallel processing with ThreadPoolExecutor
    all_results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all chunks to workers
        futures = [
            executor.submit(
                process_chunk,
                chunk=task_chunks[i],
                worker_id=i,
                client=client,
                output_prefix=args.output_prefix,
                model_name=args.model_name,
                temperature=args.temperature,
                retry_delay=args.retry_delay
            )
            for i in range(args.max_workers)
        ]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing workers"):
            all_results.update(future.result())
    
    # 5. Optional: Save combined results to a single JSON file
    if args.save_combined:
        combined_output_path = f"{args.output_prefix}combined_results.json"
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Combined results saved to: {combined_output_path}")
    
    # Final stats
    print(f"\nProcessing completed.")
    print(f"Total successful triplets: {len(all_results)}")
    print(f"Output jsonl files prefix: {args.output_prefix}worker_*.jsonl")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate relation-aware prompts via OpenAI API")
    
    # Required arguments
    parser.add_argument(
        "--openai-api-key",
        type=str,
        required=True,
        help="OpenAI API key for authentication (mandatory)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vg", "oiv6"],
        help="Target dataset to use predefined entities/predicates (choose 'vg' or 'oiv6')"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix path for output jsonl files (e.g., './DATASET/VG150/vg_relation_aware_prompt_')"
    )
    
    # Optional API configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the OpenAI model to use (default: 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for API requests (0.0 = deterministic, default: 0.0)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Delay (in seconds) before retrying after API error (default: 10)"
    )
    
    # Optional processing configuration
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Number of parallel workers (default: 30)"
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Whether to save all results into a single combined JSON file (default: False)"
    )
    
    args = parser.parse_args()
    main(args)
