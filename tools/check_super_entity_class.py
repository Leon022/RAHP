import json
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
import os


def init_openai_client(api_key: str) -> OpenAI:
    """
    Initialize OpenAI client with the provided API key.
    
    Args:
        api_key: Valid OpenAI API key for authentication.
    
    Returns:
        Initialized OpenAI client instance.
    """
    return OpenAI(api_key=api_key)


def load_json_data(file_path: str) -> dict:
    """
    Load JSON data from a specified file path.
    
    Args:
        file_path: Absolute or relative path to the JSON file.
    
    Returns:
        Parsed JSON data (dict format).
    
    Raises:
        FileNotFoundError: If the target file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found at path: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_data(data: dict, save_path: str) -> None:
    """
    Save dictionary data to a JSON file with formatted output.
    
    Args:
        data: Data to be saved (must be JSON-serializable).
        save_path: Absolute or relative path for the output JSON file.
    """
    parent_dir = os.path.dirname(save_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Data saved successfully to: {save_path}")


def build_cluster_superclass_prompt(cluster_entities: list[str]) -> str:
    """
    Build prompt to generate a superclass name for a cluster of entities.
    
    Args:
        cluster_entities: List of entities in a single cluster (e.g., ["hat", "helmet", "shirt"]).
    
    Returns:
        Formatted prompt string for OpenAI API.
    """
    prompt_template = """Task Description: You will receive a list of entities that belong to the same semantic cluster. Your job is to generate a concise, 1-3 word superclass name that accurately encapsulates all entities in the cluster. The superclass should be a general category that all entities clearly belong to.

Input: Cluster entities = {cluster_entities}

Output: Only return the 1-3 word superclass name. Do not add extra explanations or text.
    """
    return prompt_template.format(cluster_entities=str(cluster_entities)).strip()


def generate_cluster_superclass(
    cluster_entities: list[str],
    client: OpenAI,
    model_name: str,
    retry_delay: int
) -> str:
    """
    Call OpenAI API to generate a superclass name for a cluster of entities.
    
    Args:
        cluster_entities: List of entities in the cluster.
        client: Initialized OpenAI client.
        model_name: Name of the OpenAI model to use.
        retry_delay: Seconds to wait before retrying after an API error.
    
    Returns:
        Generated 1-3 word superclass name for the cluster.
    """
    prompt = build_cluster_superclass_prompt(cluster_entities)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                top_p=1.0,
                temperature=0.0
            )
            superclass = response.choices[0].message.content.strip()
            # Ensure superclass is 1-3 words (retry if not)
            if 1 <= len(superclass.split()) <= 3:
                print(f"Generated superclass for cluster: {superclass} (entities: {cluster_entities[:3]}...)")
                return superclass
            print(f"[{datetime.now()}] Superclass '{superclass}' is not 1-3 words. Retrying...")
        
        except Exception as e:
            error_msg = f"[{datetime.now()}] API error generating cluster superclass: {str(e)}. Retrying in {retry_delay}s..."
            print(error_msg)
            time.sleep(retry_delay)


def build_entity_validation_prompt(entity: str, all_superclasses: list[str]) -> str:
    """
    Build prompt to validate which superclass an entity belongs to (from existing superclasses).
    
    Args:
        entity: Single entity to validate (e.g., "hat").
        all_superclasses: List of pre-generated superclass names (e.g., ["clothing", "furniture"]).
    
    Returns:
        Formatted prompt string for OpenAI API.
    """
    prompt_template = """Task Description: You will get a single entity and a list of pre-defined superclasses. Your job is to select the ONE superclass from the list that the entity most logically and accurately belongs to.

Input:
- Entity = {entity}
- Pre-defined superclasses = {all_superclasses}

Output: Only return the selected superclass name. Do not add extra explanations or text. If none fit, return the superclass that is closest in meaning.
    """
    return prompt_template.format(
        entity=entity,
        all_superclasses=str(all_superclasses)
    ).strip()


def assign_entity_to_superclass(
    entity: str,
    all_superclasses: list[str],
    client: OpenAI,
    model_name: str,
    retry_delay: int
) -> str:
    """
    Validate and assign an entity to the correct superclass (from pre-generated superclasses).
    
    Args:
        entity: Single entity to assign (e.g., "jean").
        all_superclasses: List of pre-generated superclass names.
        client: Initialized OpenAI client.
        model_name: Name of the OpenAI model to use.
        retry_delay: Seconds to wait before retrying after an API error.
    
    Returns:
        Correct superclass name assigned to the entity.
    """
    prompt = build_entity_validation_prompt(entity, all_superclasses)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                top_p=1.0,
                temperature=0.0
            )
            assigned_superclass = response.choices[0].message.content.strip()
            # Ensure assigned superclass is in the pre-generated list (retry if not)
            if assigned_superclass in all_superclasses:
                return assigned_superclass
            print(f"[{datetime.now()}] Assigned superclass '{assigned_superclass}' not in pre-defined list. Retrying...")
        
        except Exception as e:
            error_msg = f"[{datetime.now()}] API error assigning superclass to {entity}: {str(e)}. Retrying in {retry_delay}s..."
            print(error_msg)
            time.sleep(retry_delay)


def main(args):
    # 1. Initialize OpenAI client
    client = init_openai_client(args.openai_api_key)
    
    # 2. Load input data
    print("Step 1/3: Loading input data...")
    # Load cluster-entity mapping (e.g., {"1": ["hat", "helmet"], "2": ["table", "chair"]})
    cluster_data = load_json_data(args.super_entities_path)
    cluster_entity_map = cluster_data.get("cluster_entity_groups")
    if not cluster_entity_map or not isinstance(cluster_entity_map, dict):
        raise KeyError("Super entities JSON missing 'cluster_entity_groups' (must be a dict)")
    
    # Load entity list to process (from cate_info's "ent_cate" field)
    cate_info = load_json_data(args.cate_info_path)
    if 'vg' in args.cate_info_path:
        entity_list = cate_info.get("ent_cate", [])[:-1]  # Get all entities to process   
    elif 'oiv6' in args.cate_info_path:
        entity_list = cate_info.get("obj", [])  # Get all entities to process   
    if not entity_list:
        raise KeyError("cate_info JSON missing 'ent_cate' (must be a dict of entity:predicates)")
    print(f"Loaded {len(cluster_entity_map)} clusters and {len(entity_list)} entities to process")


    # 3. Step 1: Generate superclass for each cluster
    print("\nStep 2/3: Generating superclasses for each cluster...")
    cluster_superclass_map = {}  # Key: cluster ID, Value: generated superclass
    for cluster_id, cluster_entities in tqdm(cluster_entity_map.items(), desc="Processing clusters"):
        superclass = generate_cluster_superclass(
            cluster_entities=cluster_entities,
            client=client,
            model_name=args.model_name,
            retry_delay=args.retry_delay
        )
        cluster_superclass_map[cluster_id] = superclass
    
    # Collect all unique pre-generated superclasses
    all_superclasses = list(cluster_superclass_map.values())
    print(f"\nGenerated {len(all_superclasses)} unique superclasses: {all_superclasses}")


    # 4. Step 2: Assign each entity to the correct superclass (validation)
    print("\nStep 3/3: Validating and assigning superclasses to entities...")
    entity_superclass_map = {}
    for entity in tqdm(entity_list, desc="Processing entities"):
        assigned_superclass = assign_entity_to_superclass(
            entity=entity,
            all_superclasses=all_superclasses,
            client=client,
            model_name=args.model_name,
            retry_delay=args.retry_delay
        )
        entity_superclass_map[entity] = assigned_superclass


    # 5. Save results (include both cluster-superclass and entity-superclass mappings)
    final_results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": args.model_name,
            "total_clusters": len(cluster_superclass_map),
            "total_entities": len(entity_superclass_map),
            "generated_superclasses": all_superclasses
        },
        "cluster_to_superclass": cluster_superclass_map,
        "entity_to_superclass": entity_superclass_map
    }
    
    save_json_data(final_results, args.save_path)
    print(f"\nFinal results saved to: {args.save_path}")
    print(f"Example assignments: {dict(list(entity_superclass_map.items())[:5])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cluster superclasses + assign entities to superclasses via OpenAI API"
    )
    
    # Required arguments
    parser.add_argument(
        "--openai-api-key",
        type=str,
        required=True,
        help="Valid OpenAI API key (mandatory for API access)"
    )
    parser.add_argument(
        "--super-entities-path",
        type=str,
        required=True,
        help="Path to JSON with 'cluster_entity_groups' (e.g., './DATASET/VG150/vg_super_entities.json')"
    )
    parser.add_argument(
        "--cate-info-path",
        type=str,
        required=True,
        help="Path to JSON with 'ent_cate' (entity:predicates dict, e.g., './DATASET/VG150/vg_cate_dict.json')"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save final results (e.g., './DATASET/VG150/vg_entity_superclass_final.json')"
    )
    
    # Optional API configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name to use (default: 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Seconds to wait before retrying after API errors (default: 10)"
    )
    
    args = parser.parse_args()
    main(args)
