# HAD-CSGAN
Cross-Scene Adversarial Learning With Gaussian Mixture Model for Hyperspectral Anomaly Detection
# Abstract
Most of the existing deep learning-based hyperspectral anomaly detection (HAD) methods distinguish the abnormal pixels from the surrounding background based on background models or anomaly-background separation models. However, the training samples separated from the image in unsupervised manners are usually impure, and then greatly limit the anomaly detection capability of these models. Here, we present a domain adaptation-based adversarial learning framework (termed as HAD-CSGAN) for cross-scene hyperspectral anomaly detection, which learns to reconstruct background by transferring the shared background acknowledge of related scene (source scene) to the scene without any prior information (target scene). Specifically, HAD-CSGAN is composed of two novel Gaussian mixture model (GMM)-based discriminative learning adversarial autoencoders (GMM-DLAAE). In adversarial training, the imposed prior distribution (IPD) is represented by the GMM estimated from source background to accurately characterize the real distribution because of the complexity of hyperspectral images. To further transfer the background knowledge of the source scene effectively, we introduce multi-kernel maximum mean discrepancy (MK-MMD) to restrict the domain distribution discrepancy between the source and target scenes. Additionally, an IPD corrector (called MC-FT) is used to fine-tune the mixture coefficients of GMM in target scene to make the encoded representation approximate the real distribution of target scene. The experiments conducted on five groups of datasets illustrate the superiority of the proposed HAD-CSGAN in HAD compared with twelve state-of-the-art HAD algorithms.
# Requirements
Ubuntu 20.04 cuda 11.0
Python 3.7 Pytorch 1.7
