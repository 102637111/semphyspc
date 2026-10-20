import os
import torch
import numpy as np
import json
from tools import builder
from utils import misc
from utils.logger import *
from utils.config import get_config
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 
    'wardrobe', 'xbox'
]

def load_model(args, config):
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model = base_model.cuda()
    base_model = torch.nn.DataParallel(base_model)
    base_model.eval()
    return base_model

def test_with_results(base_model, test_dataloader, config, args):
    test_pred = []
    test_label = []
    test_taxonomy = []
    test_model_ids = []
    npoints = config.npoints
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            
            points = misc.fps(points, npoints)
            logits = base_model(points)
            
            pred = logits.argmax(-1).view(-1)
            target = label.view(-1)
            
            test_pred.append(pred.detach().cpu().numpy())
            test_label.append(target.detach().cpu().numpy())
            test_taxonomy.extend([t[0] for t in taxonomy_ids])
            test_model_ids.extend([m[0] for m in model_ids])
    
    test_pred = np.concatenate(test_pred)
    test_label = np.concatenate(test_label)
    
    acc = (test_pred == test_label).sum() / float(test_label.shape[0]) * 100.
    print(f"Overall Accuracy: {acc:.4f}%")
    
    return test_pred, test_label, test_taxonomy, test_model_ids, acc

def generate_report(test_pred, test_label, taxonomy_ids, model_ids, acc, args):
    report = classification_report(test_label, test_pred, target_names=CLASS_NAMES, output_dict=True)
    
    results = {
        'overall_accuracy': acc,
        'classification_report': report,
        'predictions': [],
        'misclassified': []
    }
    
    for i in range(len(test_label)):
        is_correct = test_pred[i] == test_label[i]
        result = {
            'model_id': model_ids[i],
            'taxonomy': taxonomy_ids[i],
            'true_label': CLASS_NAMES[test_label[i]],
            'pred_label': CLASS_NAMES[test_pred[i]],
            'correct': bool(is_correct)
        }
        results['predictions'].append(result)
        if not is_correct:
            results['misclassified'].append(result)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/results.json")
    print(f"Total samples: {len(test_label)}")
    print(f"Correct: {sum(test_pred == test_label)}")
    print(f"Misclassified: {len(results['misclassified'])}")
    
    cm = confusion_matrix(test_label, test_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {args.output_dir}/confusion_matrix.png")
    
    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if (test_label == i).sum() > 0:
            class_acc[name] = (test_pred[test_label == i] == i).sum() / (test_label == i).sum() * 100.
        else:
            class_acc[name] = 0.
    
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
    print("\nPer-class accuracy:")
    for name, accuracy in sorted_acc:
        print(f"  {name}: {accuracy:.2f}%")
    
    with open(os.path.join(args.output_dir, 'per_class_accuracy.txt'), 'w') as f:
        for name, accuracy in sorted_acc:
            f.write(f"{name}: {accuracy:.2f}%\n")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--ckpts', type=str, required=True, help='checkpoint path')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='output directory')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--exp_name', type=str, default='eval', help='experiment name')
    args = parser.parse_args()
    
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.local_rank = 0
    args.experiment_path = os.path.join('./experiments', args.exp_name)
    
    config = get_config(args)
    config.dataset.test.others.bs = 16
    
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    base_model = load_model(args, config)
    
    print(f"Testing with checkpoint: {args.ckpts}")
    print(f"Config: {args.config}")
    print(f"GPU available: {args.use_gpu}")
    
    test_pred, test_label, taxonomy_ids, model_ids, acc = test_with_results(base_model, test_dataloader, config, args)
    
    generate_report(test_pred, test_label, taxonomy_ids, model_ids, acc, args)
    
    print("\n=== Evaluation complete ===")

if __name__ == '__main__':
    main()