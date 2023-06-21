# Hello
---
\section {ALGORITHM}

from an overview, our method involves adaptively creating augmentations based on the current saliency map. *** current saliency? start from beginning ***

For any augmentation, the saliency is calculated by a lightweight saliency method, in our case GradCAM, that is "unpermuted" by spreading values back to patch locations in the original image *** rephrase? ***

Since GradCAM is a coarse saliency map, the challenge here is a strategy to upsample these values.

---
\subsection {removing the network head}

\subsection {calculating receptive fields}
for an arbitrary network, this can be maintained as a dictionary:
1. removing the head can be cumbersome, its easier to register a forward hook for the layer of interest.
2. till you reach a size the final layer can ingest, you'll see size errors. these may be triggered by the layer of interest or ones below it.
at the size where a $1x1$ output is calculated, we have the min_size
3. if the errors abate only at a feature size larger than $3x3$ this may be due to normalizing layers requiring to average over >1 element. in this case use the debugger to ascertain when the layer size is 1x1
3. calculate the size at which a new element is introduced. this will be when the size = min_size + stride.

