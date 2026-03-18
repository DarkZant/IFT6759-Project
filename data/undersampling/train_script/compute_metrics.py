import numpy as np

cm = np.array([[4.7929547e+07, 3.1373000e+05, 2.4110250e+06],
 [1.2005900e+05 ,1.6908700e+05 ,7.4480000e+03],
 [9.7516400e+05 ,1.9238000e+04 ,2.0235980e+06]])

class_names = ['Background', 'Tropical Cyclone', 'Atmospheric River']

print('=' * 65)
print(f'{"Métrique":<22} {"Background":>12} {"TC":>12} {"AR":>12}')
print('=' * 65)

ious       = []
precisions = []
recalls    = []
specs      = []

for i, name in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP   # colonne i - diagonale
    FN = cm[i, :].sum() - TP   # ligne i - diagonale
    TN = cm.sum() - TP - FP - FN

    iou       = TP / (TP + FP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    recall    = TP / (TP + FN + 1e-7)
    spec      = TN / (TN + FP + 1e-7)

    ious.append(iou)
    precisions.append(precision)
    recalls.append(recall)
    specs.append(spec)

def row(label, values):
    print(f'{label:<22} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f}')

row('IoU',          ious)
row('Précision',    precisions)
row('Rappel',       recalls)
row('Spécificité',  specs)
print('=' * 65)
print(f'{"Mean IoU":<22} {np.mean(ious):>12.4f}')
print()

# Comparaison avec baseline
baseline_tc = {'iou': 0.3486, 'precision': 0.4599, 'recall': 0.5903, 'spec': 0.9961}
baseline_ar = {'iou': 0.3848, 'precision': 0.4543, 'recall': 0.7157, 'spec': 0.9491}

print('=' * 65)
print('COMPARAISON AVEC BASELINE')
print('=' * 65)
print(f'{"Métrique":<22} {"Baseline":>10} {"Top 75%":>10} {"Δ":>10}')
print('-' * 55)

metrics_tc = {'iou': ious[1], 'precision': precisions[1], 'recall': recalls[1], 'spec': specs[1]}
metrics_ar = {'iou': ious[2], 'precision': precisions[2], 'recall': recalls[2], 'spec': specs[2]}

print('--- Tropical Cyclone ---')
for k in ['iou', 'precision', 'recall', 'spec']:
    delta = metrics_tc[k] - baseline_tc[k]
    arrow = '✅' if delta > 0 else '❌'
    print(f'  {k:<20} {baseline_tc[k]:>10.4f} {metrics_tc[k]:>10.4f} {delta:>+10.4f} {arrow}')

print('--- Atmospheric River ---')
for k in ['iou', 'precision', 'recall', 'spec']:
    delta = metrics_ar[k] - baseline_ar[k]
    arrow = '✅' if delta > 0 else '❌'
    print(f'  {k:<20} {baseline_ar[k]:>10.4f} {metrics_ar[k]:>10.4f} {delta:>+10.4f} {arrow}')
