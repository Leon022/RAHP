import nltk
import numpy as np
import argparse
import json
import os
from nltk.corpus import wordnet as wn
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as dist


def download_nltk_resource(resource: str = 'wordnet') -> None:
    """
    Download required NLTK resource (e.g., WordNet) if not already present.
    
    Args:
        resource: Name of the NLTK resource to download (default: 'wordnet').
    """
    try:
        nltk.data.find(f'corpora/{resource}')
        print(f"NLTK resource '{resource}' is already available.")
    except LookupError:
        print(f"Downloading NLTK resource '{resource}'...")
        nltk.download(resource)


def load_entity_classes(dataset: str, cate_info_path: str) -> list[str]:
    """
    Load entity classes from category info JSON, with dataset-specific parsing logic.
    
    Args:
        dataset: Target dataset ('vg' or 'oiv6'), determines parsing rule.
        cate_info_path: Path to category info JSON file.
    
    Returns:
        List of entity classes for clustering.
    
    Raises:
        ValueError: If dataset is not 'vg' or 'oiv6', or required fields are missing.
        FileNotFoundError: If category info file does not exist.
    """
    # Check if file exists
    if not os.path.exists(cate_info_path):
        raise FileNotFoundError(f"Category info file not found: {cate_info_path}")
    
    # Load JSON data
    with open(cate_info_path, 'r', encoding='utf-8') as f:
        cate_info = json.load(f)
    
    # Parse entity classes based on dataset
    if dataset == 'vg':
        if "ent_cate" not in cate_info:
            raise KeyError(f"VG Category info missing required 'obj' field: {cate_info_path}")
        obj_classes = cate_info["ent_cate"]
    elif dataset == 'oiv6':
        if "obj" not in cate_info:
            raise KeyError(f"OIV6 Category info missing required 'obj' field: {cate_info_path}")
        obj_classes = cate_info["obj"]
    
    # Validate loaded entities
    if not isinstance(obj_classes, list) or len(obj_classes) == 0:
        raise ValueError(f"Loaded entity classes are empty or invalid for {dataset}")
    
    print(f"Successfully loaded {len(obj_classes)} entity classes for {dataset}")
    return obj_classes


def calculate_wordnet_similarity(word1: str, word2: str) -> float:
    """
    Calculate maximum Wu-Palmer (wup) similarity between two entities using WordNet.
    
    Args:
        word1: First entity for similarity comparison.
        word2: Second entity for similarity comparison.
    
    Returns:
        Maximum wup similarity (0.0 if no valid synsets are found for either entity).
    """
    # Get all WordNet synsets for both entities
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    
    # Return 0 if either entity has no synsets (unrecognized word)
    if not synsets1 or not synsets2:
        return 0.0
    
    # Find maximum similarity across all synset pairs
    max_similarity = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            current_sim = syn1.wup_similarity(syn2)
            if current_sim is not None and current_sim > max_similarity:
                max_similarity = current_sim
    
    return max_similarity


def build_similarity_matrix(entities: list[str]) -> np.ndarray:
    """
    Build symmetric similarity matrix for entities using WordNet similarity.
    
    Args:
        entities: List of entity classes (e.g., ['male', 'female', 'wild animal']).
    
    Returns:
        Symmetric numpy array of shape (n_entities, n_entities) with similarity scores.
    """
    num_entities = len(entities)
    similarity_matrix = np.zeros((num_entities, num_entities), dtype=np.float32)
    
    # Fill matrix (upper triangle + mirror to lower triangle for symmetry)
    for i in range(num_entities):
        for j in range(i, num_entities):
            sim_score = calculate_wordnet_similarity(entities[i], entities[j])
            similarity_matrix[i, j] = sim_score
            similarity_matrix[j, i] = sim_score
    
    return similarity_matrix


def perform_hierarchical_clustering(
    similarity_matrix: np.ndarray,
    distance_threshold: float,
    linkage_method: str = 'ward'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering on entity similarity matrix.
    
    Args:
        similarity_matrix: Symmetric similarity matrix (output of build_similarity_matrix).
        distance_threshold: Maximum distance to form a cluster (criterion='distance').
        linkage_method: Linkage method for clustering (default: 'ward'—minimizes cluster variance).
    
    Returns:
        Tuple of (linkage_matrix, cluster_assignments):
            - linkage_matrix: Hierarchical clustering linkage matrix.
            - cluster_assignments: Array where each element is the cluster ID of the corresponding entity.
    """
    # Convert similarity to distance (distance = 1 - similarity)
    distance_matrix = 1 - similarity_matrix
    # Ensure diagonal is 0 (distance from an entity to itself)
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Convert square distance matrix to condensed form (required for scipy linkage)
    condensed_distance = dist.squareform(distance_matrix)
    
    # Compute linkage matrix
    linkage_matrix = sch.linkage(condensed_distance, method=linkage_method)
    
    # Assign entities to clusters based on distance threshold
    cluster_assignments = sch.fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    return linkage_matrix, cluster_assignments


def save_cluster_results(
    dataset: str,
    entities: list[str],
    cluster_assignments: np.ndarray,
    distance_threshold: float,
    linkage_method: str,
    save_path: str
) -> None:
    """
    Save entity clustering results to JSON file with dataset context.
    
    Args:
        dataset: Dataset name ('vg' or 'oiv6') for result traceability.
        entities: List of original entity classes.
        cluster_assignments: Array of cluster IDs for each entity.
        distance_threshold: Distance threshold used for clustering.
        linkage_method: Linkage method used for clustering.
        save_path: Path to save the JSON result file.
    """
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(save_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created parent directory for output: {parent_dir}")
    
    # Organize result data
    # 1. Entity-to-cluster mapping (easy lookup)
    entity_cluster_map = {
        entity: int(cluster_id) 
        for entity, cluster_id in zip(entities, cluster_assignments)
    }
    
    # 2. Cluster-to-entities mapping (group entities by cluster)
    cluster_entity_groups = {}
    unique_clusters = np.unique(cluster_assignments)
    for cluster_id in unique_clusters:
        cluster_entities = [
            entity for entity, cid in zip(entities, cluster_assignments) 
            if cid == cluster_id
        ]
        cluster_entity_groups[int(cluster_id)] = cluster_entities
    
    # 3. Full result structure (with metadata for reproducibility)
    output_data = {
        "metadata": {
            "dataset": dataset,
            "clustering_timestamp": os.popen('date "+%Y-%m-%d %H:%M:%S"').read().strip(),
            "distance_threshold": distance_threshold,
            "linkage_method": linkage_method,
            "total_entities": len(entities),
            "total_clusters": int(len(unique_clusters))
        },
        "entity_cluster_mapping": entity_cluster_map,
        "cluster_entity_groups": cluster_entity_groups,
        "original_entities": entities
    }
    
    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nClustering results saved to: {save_path}")


def print_cluster_summary(entities: list[str], cluster_assignments: np.ndarray) -> None:
    """
    Print human-readable summary of cluster results (for immediate verification).
    
    Args:
        entities: List of original entity classes.
        cluster_assignments: Array of cluster IDs for each entity.
    """
    unique_clusters = sorted(np.unique(cluster_assignments))
    print("\n=== Entity Clustering Summary ===")
    print(f"Total entities: {len(entities)}")
    print(f"Total clusters: {len(unique_clusters)}\n")
    
    # Print each cluster and its entities
    for cluster_id in unique_clusters:
        cluster_entities = [
            entity for entity, cid in zip(entities, cluster_assignments) 
            if cid == cluster_id
        ]
        print(f"Cluster {cluster_id} ({len(cluster_entities)} entities):")
        print(f"  {', '.join(cluster_entities)}\n")


def main(args):
    # Step 1: Prepare dependencies (download WordNet if needed)
    download_nltk_resource()
    
    # Step 2: Load entity classes (dataset-specific logic)
    print(f"\nLoading entity classes for {args.dataset}...")
    obj_classes = load_entity_classes(
        dataset=args.dataset,
        cate_info_path=args.cate_info_path
    )
    
    # Step 3: Build entity similarity matrix
    print("\nCalculating WordNet similarity matrix for entities...")
    similarity_matrix = build_similarity_matrix(obj_classes)
    # Print small subset of matrix for verification
    print("Similarity Matrix (first 5x5 subset):")
    print(np.round(similarity_matrix[:5, :5], 3))  # Round to 3 decimals for readability
    
    # Step 4: Run hierarchical clustering
    print(f"\nStarting hierarchical clustering...")
    linkage_matrix, cluster_assignments = perform_hierarchical_clustering(
        similarity_matrix=similarity_matrix,
        distance_threshold=args.distance_threshold,
        linkage_method=args.linkage_method
    )
    
    # Step 5: Print summary of results
    print_cluster_summary(obj_classes, cluster_assignments)
    
    # Step 6: Save results to specified path
    save_cluster_results(
        dataset=args.dataset,
        entities=obj_classes,
        cluster_assignments=cluster_assignments,
        distance_threshold=args.distance_threshold,
        linkage_method=args.linkage_method,
        save_path=args.save_path
    )


if __name__ == "__main__":
    # Parse command-line arguments (all configurable for VG/OIV6)
    parser = argparse.ArgumentParser(
        description="Hierarchical Clustering of Entities for VG/OIV6 Datasets Using WordNet Similarity"
    )
    
    # Required arguments (dataset + data paths)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['vg', 'oiv6'],
        help="Target dataset (must be 'vg' or 'oiv6')—determines entity parsing logic"
    )
    parser.add_argument(
        "--cate-info-path",
        type=str,
        required=True,
        help="Path to category info JSON file (e.g., './vg_cate_dict.json' for VG, './oiv6_cate_dict.json' for OIV6)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save clustering results (e.g., './DATASET/VG/vg_entity_clusters.json' or './DATASET/OIV6/oiv6_entity_clusters.json')"
    )
    
    # Clustering hyperparameters (optional with sensible defaults)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Maximum distance to form clusters (smaller = more clusters; default: 0.5)"
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default='ward',
        choices=['ward', 'single', 'complete', 'average'],
        help="Linkage method for clustering (default: 'ward'—balances cluster size)"
    )
    
    args = parser.parse_args()
    main(args)
