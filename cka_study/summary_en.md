# ImageNet Linear CKA: measured results and interpretation

Data: ILSVRC/imagenet-1k, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`, validation, one image per class (1000 total).

Selection: first decodable, byte-unique image per class in the fixed stream order; not uniform random sampling.

Models: three IMAGENET1K_V1 CNNs. Representation: full-channel GAP of seven raw convolution outputs. Estimator: ordinary centered Linear CKA.


## vgg16 and resnet50

Maximum: features.0 (L01) versus conv1 (L01), CKA=0.9957.

Same-slot early (L01-L02), middle (L03-L05), and late (L06-L07) means are 0.7369, 0.6028, and 0.3503; the early mean is higher than the late mean.

L07 versus L07 is 0.1285. The same-slot sequence increases at 2 transitions, so the group trend is not a monotonic layer-by-layer decrease.

Cross-slot example: vgg16 L07 versus resnet50 L05 gives 0.3781, compared with 0.1285 at the matching slot. Slot-only comparisons miss some relationships; this does not establish semantic alignment.

Maximum absolute changes from the 1000-image matrix are 0.0619 and 0.0370 for 250 and 500 images. Class coverage changes too.

7/7 same-slot scores exceed the maximum of 10 mismatches. Gaps from mismatch means range from 0.1006 to 0.9923; this is not a formal significance test.


## vgg16 and mobilenet_v2

Maximum: features.0 (L01) versus features.0.0 (L01), CKA=0.9574.

Same-slot early (L01-L02), middle (L03-L05), and late (L06-L07) means are 0.8951, 0.5769, and 0.4347; the early mean is higher than the late mean.

L07 versus L07 is 0.2067. The same-slot sequence increases at 1 transitions, so the group trend is not a monotonic layer-by-layer decrease.

Cross-slot example: vgg16 L03 versus mobilenet_v2 L02 gives 0.8321, compared with 0.5787 at the matching slot. Slot-only comparisons miss some relationships; this does not establish semantic alignment.

Maximum absolute changes from the 1000-image matrix are 0.0541 and 0.0328 for 250 and 500 images. Class coverage changes too.

7/7 same-slot scores exceed the maximum of 10 mismatches. Gaps from mismatch means range from 0.1793 to 0.9539; this is not a formal significance test.


## resnet50 and mobilenet_v2

Maximum: conv1 (L01) versus features.0.0 (L01), CKA=0.9763.

Same-slot early (L01-L02), middle (L03-L05), and late (L06-L07) means are 0.8544, 0.6651, and 0.5545; the early mean is higher than the late mean.

L07 versus L07 is 0.5346. The same-slot sequence increases at 1 transitions, so the group trend is not a monotonic layer-by-layer decrease.

Cross-slot example: resnet50 L03 versus mobilenet_v2 L02 gives 0.8679, compared with 0.6350 at the matching slot. Slot-only comparisons miss some relationships; this does not establish semantic alignment.

Maximum absolute changes from the 1000-image matrix are 0.1206 and 0.0484 for 250 and 500 images. Class coverage changes too.

7/7 same-slot scores exceed the maximum of 10 mismatches. Gaps from mismatch means range from 0.4123 to 0.9728; this is not a formal significance test.


## Reading these findings

A CKA near 1 describes similar sample-relation structure, not percent classification accuracy or a percentage of identical neurons.

All three pairs attain their matrix maximum at L01-L01: True. Intermediate increases and cross-slot peaks prevent a universal monotonic-depth claim.

Early raw convolution is approximately linear, and GAP removes spatial arrangement. Global color/brightness statistics may therefore drive high early similarity. This is an interpretation hypothesis, not a tested color-ablation result or proof of identical edge detectors.

Nine real early/middle/late pairs passed equivalent feature-space recomputation, maximum error 1.89e-15. This checks arithmetic on saved features, not an independent CNN replay.


## Scope and limitations

The quota sample has one image per class and cannot estimate within-class variation. Byte-duplicate exclusion is not near-duplicate removal.

The compared slots have different operation roles, channel counts, and receptive fields; ResNet taps precede BatchNorm and residual addition.

GAP discards spatial layout. Ordinary CKA can depend on dimension, sample size, and dominant variations. High mismatches or large subset changes call for more data and independent/debiased analyses.

High CKA is not accuracy, fairness, or shared causal evidence; low CKA does not imply useless features.


## Relationship to the perturbation study

The CKA matrices quantify representation geometry. The following insertion/deletion study tests fixed-class spatial rankings. Their agreement or disagreement must be interpreted as complementary evidence, not an established causal relationship.