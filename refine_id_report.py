from pathlib import Path
import json
import ast

p=Path(__file__).resolve().parent/'A2-cka-study.ipynb'
n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    if isinstance(c['source'],list):c['source']=''.join(c['source'])
s=n['cells'][2]['source']
s=s.replace('仅下载与前向；不训练CNN，不需要Lucent或Captum。','CNN参数保持冻结：CKA前向提取，Grad-CAM计算输入相关梯度；不训练CNN。')
n['cells'][2]['source']=s
s=n['cells'][22]['source']
if 'execution_audit_rows' not in s:
    s=s.replace('display(id_summary.round(5))', '''execution_audit_rows=[]
for model_name in SPECS:
    audit=json.loads((ID_DIR/f'{model_name}_id.json').read_text(encoding='utf-8'))
    execution_audit_rows.append({'model':model_name,'original_run_seconds':audit['seconds'],
        'non_inplace_forward_max_error':audit['relu_forward_equivalence_max_error'],
        'backward_warning_count':len(audit['backward_warnings']),
        'all_zero_maps':audit['all_zero_maps'],'endpoint_checks':audit['endpoint_checks_passed']})
display(pd.DataFrame(execution_audit_rows))
print('VGG16 backward records a CUDA adaptive-average-pooling nondeterminism warning; see the stated warn-only protocol.')
display(id_summary.round(5))''')
n['cells'][22]['source']=s
s=n['cells'][25]['source']
if 'all_layers=id_summary' not in s:
    s=s.replace('for name in SPECS:', '''all_layers=id_summary[id_summary['method']!='random']
id_lines.extend(['\\n## 综合发现',
    f"21个模型—层组合中，{int((all_layers.ins_gain_ci_low>0).sum())}个Insertion改进区间完全高于0；Deletion对应数量为{int((all_layers.del_gain_ci_low>0).sum())}，另有{int((all_layers.del_gain_ci_high<0).sum())}个Deletion改进区间完全低于0。所有区间均为未经多重比较校正的探索性区间。",
    '三种模型的L07均取得各自最高Insertion AUC，但没有层在Deletion上获得完全为正的改进区间。因此只能说部分空间排序有助于分数恢复，不能据此宣称其在两种扰动评估下都可靠。',
    '随机散点遮挡可能比连续区域遮挡更易破坏纹理或使输入偏离自然图像分布；这是一种解释假设，尚未用其他基线或块状遮挡验证。'])
for name in SPECS:''',1)
    s=s.replace("f'七层中共{int(table.all_zero_maps.sum())}张全零CAM；它们保留在评估中，并已检查与随机排序曲线一致。'", "f'Deletion最优采样层相对随机的平均改进为{best_del.deletion_gain_vs_random:.4f}，95%区间[{best_del.del_gain_ci_low:.4f}, {best_del.del_gain_ci_high:.4f}]；正值才表示优于随机。',\n        f'七层中共{int(table.all_zero_maps.sum())}张全零CAM；它们保留在评估中，并已检查与随机排序曲线一致。'")
n['cells'][25]['source']=s
for c in n['cells']:
    if c['cell_type']=='code':ast.parse(c['source'])
p.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
print('Refined negative-result interpretation and embedded execution audit.')
