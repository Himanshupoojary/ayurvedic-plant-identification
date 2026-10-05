# Leaf Branch Architecture

## Design decision

The target user is a normal person taking a plant photo in the wild.

Therefore, the system should not require the user to crop or manually isolate:
- veins
- texture
- leaf edge
- shape

Each model receives the **same complete leaf image**.

## Leaf ensemble

```text
                         LEAF IMAGE
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        MobileNetV2        ResNet50       EfficientNetB0
             |                |                |
             v                v                v
       probability       probability       probability
          vector             vector            vector
             |                |                |
             +----------------+----------------+
                              |
                       probability fusion
                              |
                              v
                     FINAL LEAF PREDICTION
```

## Current fusion

Simple probability averaging:

```text
P_final = (P1 + P2 + P3) / 3
```

This gives us a clean baseline.

## Later improvement

We can evaluate validation accuracy and replace equal averaging with:

```text
P_final = w1*P1 + w2*P2 + w3*P3
```

where the weights are learned or derived from validation performance.

A later multimodal version can connect this leaf branch to flower/fruit/stem/root branches.
