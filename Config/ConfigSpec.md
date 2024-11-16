# Modules Config Specification

> Last update: 2024 Jun 13

## Motion Model

- Ground Truth + Random Noise

  ```yaml
  motion:
    type: "GTwithNoise"
    args:
      noise_std: 0.05
  ```

- TartanVO Motion Net

  ```yaml
  motion:
    type: "TartanVO"
    args:
      weight: "./Model/MACVO_posenet.pkl"
      eval_mode: true
  ```

## Key point selector

- Sparse Gradient

  ```yaml
  type: "RandGradientSparse"
  args:
    mask_width: 16 # Margin width to not select KP from
    grad_std: 2.0 # Select kp only on grad>+2std position
    nms_size: 15 # NMS filter used to ensure sparsity
    fix_seed: 1984 # Fixed random seed for experiment
  ```

- Grid

  ```yaml
  type: "Grid"
  args:
    mask_width: 32 # Margin width to not select KP from
  ```

- Covariance Aware

  ```yaml
  type: CovAwareSelector
  args:
    mask_width: 32
    max_depth: 100.
    kernel_size: 7
    device: cuda
    max_depth_cov: 1e3
    max_match_cov: 1e3
  ```

## Covariance Models

### Observation Covariance

- Naive Covariance (Depth Only)

  ```yaml
  type: NaiveCovariance
  args:
    depth_cov_min: 0.05
    depth_cov_max: 1.
  ```

- Depth Only

  ```yaml
  type: DepthCovariance
  args:
  ```

- Match Only

  ```yaml
  type: MatchCovariance
  args:
    match_cov_default: 0.25 # default match cov when not available (when KP is just created)
    kernel_size: 31 # Local patch size
    min_flow_cov: 0.16
  ```

- Depth + Match Covariance

  ```yaml
  type: GaussianMixtureCovariance
  args: # Same meaning as MatchOnly
    match_cov_default: 0.25
    kernel_size: 31
    min_flow_cov: 0.16
    min_depth_cov: 0.05
  ```

### Point Covariance

- Direct Covariance

- Weighted Average

## Depth

- TartanVO

  ```yaml
  depth:
    type: "TartanVO"
    args:
      weight: "./Model/TartanVO_depth_cov.pth" # Checkpoint path
      eval_mode: true # If in training/eval mode
      usegtcov: true # use RAFT-like estimated cov if false
      device: "cuda" # must be "cuda"
      cov_mode: Est # Est | GT | None
  ```

## Outlier Rejection

- FilterCompose

  ```yaml
  outlier:
    type: FilterCompose
    args:
      filter_args:
        - Subfilters to compose...
  ```

- Covariance Sanity (filter obs with `nan` and `inf` cov)

  ```yaml
  type: CovarianceSanityFilter
  args:
  ```

- Depth Filter

  ```yaml
  type: DepthFilter
  args:
    min_depth: 0.05 # Minimum depth to accept
    max_depth: 20.0 # Prefered maximum depth to accept
    expect_num: 12
    # NOTE: actual maximum depth is 100 meter. The filter will increase
    # max_depth if less than expect_num kps are remain after filtering.
  ```

## Match

- TartanVO Match (no Covariance)

  ```yaml
  type: "TartanVO"
  args:
    weight: "./Model/MACVO_posenet.pkl"
    eval_mode: true
    cov_mode: None
  ```

- TartanVO Match (GT Covariance)

  ```yaml
  type: "TartanVO"
  args:
    weight: "./Model/MACVO_posenet.pkl"
    eval_mode: true
    cov_mode: GT
  ```

- FlowFormer Match

  ```yaml
  type: "Flowformer"
  args:
    weight: "./Model/Flowformer_sintel.pth"
    eval_mode: true
    device: cuda
  ```

- TartanVO + Estimated Cov Match

  ```yaml
  type: "TartanVO"
  args:
    weight: "./Model/205001_RAFTCov_b.pth"
    eval_mode: true
    cov_mode: Est
    cfg:
      decoder: raft
      dim: 64
      dropout: 0.1
      num_heads: 4
      mixtures: 4
      gru_iters: 12
      kernel_size: 3
  ```

- FlowFormer + Estimated Cov Match

  ```yaml
  type: "FlowformerCov"
  args:
    weight: "./Model/MACVO_FrontendCov.pth"
    eval_mode: true
    device: cuda
    output_cov: false
  ```

## Frontend

- FlowFormerCov Frontend

  ```yaml
  type: FlowFormerCovFrontend
  args:
    device: *device
    eval_mode: true | false
    weight: ./Model/MACVO_FrontendCov.pth
    use_jit: true | false
    enforce_positive_disparity: true | false
  ```

- Compose Frontend

  ```yaml
  type: FrontendCompose
  args:
    depth:
      # Depth Config Here
      # type: ...
      # args:
      #   ...
    match:
      # Match Config Here
      # type: ...
      # args:
      #   ...
  ```
