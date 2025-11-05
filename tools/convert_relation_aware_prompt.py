import jsonlines
import json
import re
import argparse
from tqdm import tqdm


def extract_region_descriptions(text):
    text = text.replace('*', '')
    
    pattern = r'Region Descriptions:\s*(.*)$'
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if not match:
        return []
    
    descriptions_text = match.group(1)
    lines = descriptions_text.split('\n')
    description_list = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue  
        desc_match = re.search(r'“([^”]+)”', line_stripped)
        if desc_match:
            description_list.append(desc_match.group(1).strip())
    
    return description_list


def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, "r") as f:
        for line in f:
            data.append(line)
    return data


def main(args):
    if args.dataset == 'vg':
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
    elif args.dataset == 'oiv6':
        super_entities =  ['male', 'female', 'children', 'head feature', 'limb feature','torso feature', 'accessorie', 'mammal', 'bird', 'reptile', 'insect', 'marine animal','bike', 'ground vehicle', 'watercraft', 'aircraft', 'vehicle part item', 'ball-relatedsport item', 'water sport item', 'winter sportitem', 'seating furniture', 'table furniture','storage furniture', 'bedding', 'upper bodyclothing', 'lower body clothing', 'footwear', 'fruit', 'vegetable', 'prepared food', 'beverage', 'appliance', 'utensil', 'decorative item','textile', 'hand tool', 'power tool', 'kitchentool', 'personal electronic', 'home electronic', 'office electronic', 'land vehicle', 'watervehicle', 'air vehicle', 'string instrument','wind instrument', 'percussion instrument', 'firearm', 'container', 'toy', 'stationery', 'landscape', 'urban feature']
        predicates = [
            "at",
            "holds",
            "wears",
            "surf",
            "hang",
            "drink",
            "holding hands",
            "on",
            "ride",
            "dance",
            "skateboard",
            "catch",
            "highfive",
            "inside of",
            "eat",
            "cut",
            "contain",
            "handshake",
            "kiss",
            "talk on phone",
            "interacts with",
            "under",
            "hug",
            "throw",
            "hits",
            "snowboard",
            "kick",
            "ski",
            "plays",
            "read"
        ]

    relation_result = {}
    for worker_id in range(args.num_workers):
        input_path = f"{args.input_prefix}worker_{worker_id}.jsonl"
        print(f"Loading data from: {input_path}")
        
        data = load_jsonl(input_path)
        for item in tqdm(data, desc=f"Processing worker {worker_id}"):
            triplet = item['triplet']
            if triplet not in relation_result:
                relation_result[triplet] = extract_region_descriptions(item['rel_text'])
                if not relation_result[triplet] and args.debug:
                    print(f"Warning: Empty descriptions for triplet {triplet}")
                    import ipdb; ipdb.set_trace()

    if args.check_complete:
        for sub in super_entities:
            for obj in super_entities:
                for pred in predicates:
                    triplet = f"{sub}_{pred}_{obj}"
                    if triplet not in relation_result:
                        print(f"Missing triplet: {triplet}")
                        if args.debug:
                            import ipdb; ipdb.set_trace()

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(relation_result, f, ensure_ascii=False, indent=2)
    print(f"Result saved to: {args.output_path}")
    print(f"Total unique triplets: {len(relation_result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract region descriptions from relation prompts")
    
    parser.add_argument("--dataset", 
                       type=str, 
                       default="vg",
                       help="dataset name")
    parser.add_argument("--input_prefix", 
                       type=str, 
                       default="./DATASET/VG150/vg_relation_aware_prompt_",
                       help="Prefix path for input jsonl files (before 'worker_{i}.jsonl')")
    parser.add_argument("--num_workers", 
                       type=int, 
                       default=30, 
                       help="Number of worker files to process")
    
    parser.add_argument("--output_path", 
                       type=str, 
                       default="./DATASET/VG150/vg_relation_aware_prompts.json",
                       help="Path to save the output JSON file")
    
    parser.add_argument("--debug", 
                       action="store_true", 
                       help="Enable debug mode (break on empty descriptions)")
    parser.add_argument("--check_complete", 
                       action="store_true", 
                       help="Check if all possible triplets are present")
    
    args = parser.parse_args()
    main(args)
