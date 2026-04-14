# Baseline Fairness Documentation

This document provides detailed records of baseline modifications performed in this work to ensure fair comparison when illumination or appearance-related modeling modules are removed. These records are provided to ensure transparency, reproducibility, and traceability of the modified baseline configurations.

## 1. Overview

Some baseline methods used in our experiments include illumination-dependent or appearance-adaptive modeling components. Since the primary objective of this work is to evaluate occlusion-aware reconstruction performance rather than illumination modeling capability, illumination-related modules were disabled in selected baselines.

After disabling these modules, all modified baselines were retrained from scratch under identical experimental conditions.

## 2. Modified Baseline Methods

### 2.1 WildGaussians
#### Original Functionality
WildGaussians includes an appearance embedding module designed to model view-dependent color variations caused by illumination changes.
#### Removed Component
The following module was disabled: Appearance embedding branch responsible for view-dependent color modeling.

The appearance embedding mechanism was disabled while preserving:Gaussian representation, Rendering pipeline and Geometry optimization process.
No structural changes were applied to the Gaussian parameterization.

#### Change in the Code

- appearance_enabled: false (in the default.yml and phototourism.yml)
- disable class AppearanceModel and class EmbeddingModel (in the method.py)

#### Retained Components

The following components remained unchanged: Geometry optimization module, Gaussian rendering module and Loss computation pipeline.

### 2.2 Gaussian in the Wild (GS-W)

#### Original Functionality
Gaussian in the Wild (GS-W) includes a dynamic appearance modeling branch designed to capture illumination changes and environmental variations.

#### Removed Components
The following dynamic appearance components were removed:Projection Feature Map, K dynamic feature maps, Adaptive sampling parameters related to dynamic appearance, Dynamic appearance feature fusion branch and Dynamic feature dropout mechanism.

#### Change in the Code
The dynamic appearance branch was disabled while maintaining the remaining intrinsic appearance pathway.

- self.use_okmap=False (in the /arguments/__init__.py)
- self.use_kmap_pjmap=False (in the /arguments/__init__.py)

Disabling these two switches prevents the execution of the dynamic appearance pipeline. Consequently, the following components are automatically deactivated:
 
- Adaptive sampling parameters (box_coord in the gaussian_model.py)  
- Dynamic appearance feature extraction (_point_features in the gaussian_model.py)  

After this modification, only the intrinsic appearance features (_features_intrinsic) are used for color prediction, ensuring that illumination-dependent modeling is removed while preserving the core Gaussian representation and rendering pipeline.
#### Retained Components
After removal, only the intrinsic appearance branch was retained for color prediction. This preserved: Static Gaussian geometry, Intrinsic color prediction pipeline and Core rendering mechanism

## 3. Retraining Protocol
All modified baseline models were retrained from scratch after disabling illumination-related components.

#### Training Procedure
The following protocol was applied consistently across all modified baselines:
- Training initialized from scratch
- Default model initialization from official repositories
- Same dataset splits as used in our proposed method
- Identical training images
- Identical evaluation datasets

#### Training Configuration
The training schedules followed the default settings provided in the official implementations.
- Optimizer: Adam
- Learning rate: Default repository value
- Training iterations: Same iterations as used in our proposed method
- Loss functions: Same as original implementation
- No additional loss terms introduced

#### Hyperparameter Policy
Default hyperparameters from original repositories were preserved and no additional manual tuning was introduced.
This ensures that performance differences are not caused by hyperparameter manipulation.

## 4. Summary
This documentation provides complete transparency regarding:

- Removed appearance-related components
- Retained model structures
- Retraining procedures

These records ensure that the modified baselines used in this work can be reproduced and verified by independent researchers.