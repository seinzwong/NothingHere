# Grad-CAM insertion/deletion: measured results and interpretation

First 100 records of the fixed ImageNet subset, with no correctness filtering. Each model/image fixes its original Top-1 target across all seven layers.

ImageNet mean-color baseline; 21 sample points and probability AUC. Higher insertion and lower deletion AUC are preferred.


## Combined finding

Across 21 model-layer combinations, 17 insertion gain intervals are entirely above zero, compared with 0 deletion intervals. Another 16 deletion intervals are entirely negative. All are exploratory, unadjusted intervals.

L07 has the highest insertion AUC within each model, but no layer has an entirely positive deletion gain interval. Some rankings support score recovery, but these results do not establish effectiveness under both perturbations.

Scattered random masking may disrupt texture or push inputs off the natural-image distribution more than spatially contiguous masking. This is an explanatory hypothesis, not a verified mechanism; other baselines and patch-based interventions were not tested.


## vgg16

Random-order mean insertion/deletion AUC: 0.1054/0.1014.

Highest insertion AUC: L07 (0.3096); lowest deletion AUC: L04 (0.1327).

4/0 layers have paired insertion/deletion gain intervals entirely above zero. These are exploratory, unadjusted 95% intervals.

The best sampled deletion layer has mean gain -0.0313, 95% interval [-0.0555, -0.0072]; only positive gain favors CAM over random.

Total all-zero maps across seven layers: 2. They remain in the evaluation and are checked against their identical random-order curves.


## resnet50

Random-order mean insertion/deletion AUC: 0.1596/0.1605.

Highest insertion AUC: L07 (0.5950); lowest deletion AUC: L02 (0.1426).

6/0 layers have paired insertion/deletion gain intervals entirely above zero. These are exploratory, unadjusted 95% intervals.

The best sampled deletion layer has mean gain 0.0179, 95% interval [-0.0097, 0.0441]; only positive gain favors CAM over random.

Total all-zero maps across seven layers: 0. They remain in the evaluation and are checked against their identical random-order curves.


## mobilenet_v2

Random-order mean insertion/deletion AUC: 0.1146/0.1185.

Highest insertion AUC: L07 (0.5189); lowest deletion AUC: L03 (0.1397).

7/0 layers have paired insertion/deletion gain intervals entirely above zero. These are exploratory, unadjusted 95% intervals.

The best sampled deletion layer has mean gain -0.0212, 95% interval [-0.0468, 0.0034]; only positive gain favors CAM over random.

Total all-zero maps across seven layers: 6. They remain in the evaluation and are checked against their identical random-order curves.


## Interpretation and limits

1. Positive gains indicate faster score recovery/suppression relative to the chosen random order under this protocol, not improved accuracy.

2. Report insertion and deletion separately when they select different layers; do not cherry-pick the favorable metric.

3. Code checks fixed targets, original/baseline endpoints, pixel counts, AUC direction, non-inplace forward equivalence, and zero-map behavior.

4. One mean-color baseline, 100 classes with one image each, 20 intervals, and one random order per image do not cover other baselines, step sizes, or all random orders. Bootstrap intervals omit those uncertainties.

5. Masked images may be out of distribution. Calibration affects probabilities and model Top-1 targets may differ; these AUCs are not an accuracy ranking.

6. Similar representation geometry need not imply identical attribution or perturbation performance. No causal link between CKA and AUC is established here.