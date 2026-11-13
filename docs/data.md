Each image shows a 3D scene with simple objects (cubes, spheres, cylinders; 8 colours, 2 sizes, 2 materials: rubber or metal). Each caption is a statement about the scene that is either TRUE or FALSE for its image. Your task is to predict the probability that a caption is true.

### Files

-   train/, val/, test/ - images (240x160 JPEG). An image listed in train.csv is in train/, and so on.
-   train.csv, val.csv - captions with labels.
-   test.csv - captions without labels: predict these.
-   sample\_submission.csv - a submission in the correct format (id, label) - (id of sample, prob).

| Split | Images | Captions |
| --- | --- | --- |
| train | 5,000 |  |
| val | 1,000 | <…> |
| test | 6,000 | <…> |

### Columns

-   id - unique row id (used in the submission)
-   image - image file name
-   caption - the statement about the image
-   level - difficulty level of the caption type, 1 (simple) to 5 (compositional)
-   category - caption type (see below)
-   pair\_id - (train/val only) links a caption to its "twin": a caption for the same image that differs in one detail (a colour, a number, left/right, "a"/"no", …). Exactly one caption of each pair is true.
-   label - (train/val only) 1 = the caption is true for the image, 0 = false

test.csv has id, image, caption, level, category (no pair\_id, no label).

### Caption categories

| Level | Category | Example |
| --- | --- | --- |
| 1 | attribute | a large red metal cube |
| 2 | counting | there are exactly three cylinders |
| 3 | binding | a red cube and a blue sphere |
| 3 | negation | there is no gray cylinder |
| 4 | position | the red cube is on the left side of the image |
| 4 | spatial | a red cube to the left of the blue sphere |
| 5 | spatial\_counting | there are exactly two objects behind the green sphere |
| 5 | spatial\_negation | there is no sphere to the right of the red cube |
| 5 | comparison | there are more cubes than spheres |

Labels are balanced: 50% true, 50% false in every category. There are some of the test captions use phrasings (word orders) that DO NOT appear in train or val.