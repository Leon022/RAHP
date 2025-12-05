import os
import json
import torch
import argparse
import torch.nn as nn
from clip import clip
from tqdm import tqdm


def load_clip_to_cpu(visual_backbone: str):
    if visual_backbone not in clip._MODELS:
        raise ValueError(
            f"不支持的CLIP backbone: {visual_backbone}\n可选值: {list(clip._MODELS.keys())}"
        )
    
    model_url = clip._MODELS[visual_backbone]
    cache_dir = os.path.expanduser("~/.cache/clip")
    model_path = clip._download(model_url, cache_dir)
    
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())
    return model.cpu()


class TextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model.float()
        self.dtype = self.clip_model.dtype
        
        self.token_embedding = self.clip_model.token_embedding
        self.positional_embedding = self.clip_model.positional_embedding
        self.transformer = self.clip_model.transformer
        self.ln_final = self.clip_model.ln_final
        self.text_projection = self.clip_model.text_projection

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  #  [n_ctx, batch_size, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  #  [batch_size, n_ctx, d_model]
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def load_categorical_clip_text_embedding(dataset_name: str, dataset_dir: str):
    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(f"Not found dataset dir: {dataset_dir}")
    
    if dataset_name == "VG":
        cate_info_path = os.path.join(dataset_dir, "VG150", "vg_cate_dict.json")
        prompts_path = os.path.join(dataset_dir, "VG150", "vg_relation_aware_prompts.json")
    elif dataset_name == "OIV6":
        cate_info_path = os.path.join(dataset_dir, "Openimage V6", "oiv6_cate_dict.json")
        prompts_path = os.path.join(dataset_dir, "Openimage V6", "oiv6_relation_aware_prompts.json")
    else:
        raise ValueError(f"Not match: {dataset_name}\n in 'VG', 'OIV6'")
    
    try:
        with open(cate_info_path, "r", encoding="utf-8") as f:
            cate_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"类别信息文件缺失: {cate_info_path}")

    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            relation_aware_prompts = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"关系提示词文件缺失: {prompts_path}")
    
    if dataset_name == "VG":
        obj_classes = cate_info["ent_cate"][:-1]  
        rel_classes = cate_info["pred_cate"]   
        super_obj_classes = cate_info.get("super_ent_cate", [])
    else:  # OIV6
        obj_classes = cate_info["obj"]
        rel_classes = cate_info["rel"]
        super_obj_classes = cate_info.get("super_obj", [])
    
    return obj_classes, super_obj_classes, rel_classes, relation_aware_prompts



def build_batch_text_embedding(texts: list, text_model: TextEncoder) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = text_model.to(device).eval()  
 
    text_tokens = clip.tokenize(texts).long().to(device)
    

    with torch.no_grad():
        text_embeddings = text_model(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True) 
    # avg_embedding = text_embeddings.mean(dim=0)
    # avg_embedding /= avg_embedding.norm() 
    
    return text_embeddings.cpu(), text_tokens.cpu()


def generate_relation_aware_weight(
    rel_classes: list,
    obj_classes: list,
    relation_aware_prompts: dict,
    text_encoder: TextEncoder
) -> list:
    weight_list = []
    for rel in tqdm(rel_classes):
        rel_weight = []
        for sub in obj_classes:
            for obj in obj_classes:
                if rel == '__background__':
                    embed, _ = build_batch_text_embedding([f"There is no relation between {sub} and {obj}"], text_encoder)
                else:
                    triplet_key = f"{sub}_{rel}_{obj}"
                    if triplet_key not in relation_aware_prompts:
                        raise KeyError(f"Miss relation triplet: {triplet_key}")
                    
                    prompts = relation_aware_prompts[triplet_key]
                    embed, _ = build_batch_text_embedding(prompts, text_encoder)
                rel_weight.append(embed)
        weight_list.append(rel_weight)
    print(f"[INFO] Generate {len(rel_classes)} relation-aware embeddings...")
    return weight_list



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CLIP-based relation embedding generation")
    
    # 核心参数（必选/可选）
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=True, 
        choices=["VG", "OIV6"],
        help="数据集名称（支持VG/OIV6）"
    )
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        default=None,
        help="数据集根目录路径（优先使用此参数，未指定则读取环境变量DATASET_DIR，默认./DATASET）"
    )
    parser.add_argument(
        "--clip-backbone", 
        type=str, 
        default="ViT-B/32",
        choices=list(clip._MODELS.keys()),
        help="CLIP模型backbone（默认ViT-B/32）"
    )
    parser.add_argument(
        "--save-path", 
        type=str, 
        default=None,
        help="嵌入文件保存路径（未指定则自动生成：数据集目录/[dataset_name]_relation_aware_embedding.pt）"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = os.environ.get("DATASET_DIR", "./DATASET")

    
    print(f"[INFO] Load CLIP MODEL: {args.clip_backbone}")
    clip_model = load_clip_to_cpu(args.clip_backbone)
    text_encoder = TextEncoder(clip_model)

    # 注意：这里使用超类替换原先的entity类别
    # 函数返回值的顺序为：obj_classes, super_classes, rel_classes, relation_aware_prompts
    # 通过将第一个返回值赋值给下划线(_)，我们忽略原始entity类别信息
    # 将第二个返回值(super_classes)作为对象类别使用
    print(f"[INFO] Load {args.dataset_name} dataset...")
    _, obj_classes, rel_classes, relation_aware_prompts = load_categorical_clip_text_embedding(
        dataset_name=args.dataset_name,
        dataset_dir=dataset_dir
    )
    
    print(f"[INFO] Start generate relation embedding...")
    relation_aware_weights = generate_relation_aware_weight(
        rel_classes=rel_classes,
        obj_classes=obj_classes,
        relation_aware_prompts=relation_aware_prompts,
        text_encoder=text_encoder
    )

    if args.save_path:
        save_path = args.save_path
    else:
        save_dir = os.path.join(dataset_dir, args.dataset_name)
        os.makedirs(save_dir, exist_ok=True) 
        save_path = os.path.join(save_dir, args.dataset_name, f"{args.dataset_name}_relation_aware_embedding.pt")
    
    torch.save(relation_aware_weights, save_path)
    print(f"[INFO] Finish saved: {save_path}")


if __name__ == "__main__":
    main()
