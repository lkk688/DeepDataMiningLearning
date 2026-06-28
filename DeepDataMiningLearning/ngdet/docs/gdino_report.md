# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | mixed | kitti | waymo | nuimages |
|---|---|---|---|---|
| GDINO-tiny zero-shot | 0.123 | 0.118 | 0.173 | 0.161 |
| GDINO-tiny fine-tuned | 0.295 | 0.345 | 0.272 | 0.330 |

## Summary matrix — AP50

| model \ dataset | mixed | kitti | waymo | nuimages |
|---|---|---|---|---|
| GDINO-tiny zero-shot | 0.224 | 0.239 | 0.308 | 0.263 |
| GDINO-tiny fine-tuned | 0.489 | 0.552 | 0.462 | 0.539 |

## Full COCO tables (per run)

### GDINO-tiny zero-shot  ×  mixed  (553 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.123 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.224 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.113 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.016 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.130 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.243 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.078 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.158 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.197 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.043 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.207 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.355 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.034 |
| person | 0.307 |
| cyclist | 0.028 |

### GDINO-tiny zero-shot  ×  kitti  (200 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.118 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.239 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.102 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.035 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.126 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.184 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.103 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.199 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.208 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.142 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.210 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.259 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.046 |
| person | 0.260 |
| cyclist | 0.047 |

### GDINO-tiny zero-shot  ×  waymo  (186 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.173 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.308 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.164 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.015 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.190 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.345 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.034 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.135 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.205 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.038 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.237 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.392 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.026 |
| person | 0.320 |
| cyclist | 0.000 |

### GDINO-tiny zero-shot  ×  nuimages  (167 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.161 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.263 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.159 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.113 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.183 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.282 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.138 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.249 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.261 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.174 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.266 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.392 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.062 |
| person | 0.368 |
| cyclist | 0.053 |

### GDINO-tiny fine-tuned  ×  mixed  (553 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.295 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.489 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.314 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.061 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.322 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.512 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.155 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.344 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.430 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.220 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.459 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.637 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.341 |
| person | 0.243 |
| cyclist | 0.302 |

### GDINO-tiny fine-tuned  ×  kitti  (200 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.345 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.552 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.380 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.155 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.349 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.513 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.198 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.462 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.482 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.481 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.613 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.499 |
| person | 0.246 |
| cyclist | 0.291 |

### GDINO-tiny fine-tuned  ×  waymo  (186 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.272 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.462 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.275 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.052 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.331 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.545 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.041 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.231 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.389 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.159 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.476 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.641 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.303 |
| person | 0.242 |
| cyclist | 0.000 |

### GDINO-tiny fine-tuned  ×  nuimages  (167 imgs, 0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.330 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.539 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.346 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.165 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.318 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.499 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.219 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.427 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.455 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.259 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.448 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.598 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.373 |
| person | 0.263 |
| cyclist | 0.354 |
