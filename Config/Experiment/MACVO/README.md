# Configuration Description

## Frontend
```
type: CUDAGraph_FlowFormerCovFrontend
args:
    device: *device
    eval_mode: true
    weight: ./Model/120000_Flowformer_bf16.pth
    enforce_positive_disparity: false
    dtype: bf16
    max_flow: -1
```

## Depth Estimator

```
type: FlowFormerDepth
args:
    weight      : ./Model/MACVO_FrontendCov.pth
    eval_mode   : true | false
    cov_mode    : Est
    device      : cuda | cpu | cuda:<idx>
```

```
type: FlowFormerDepth
args:
    weight      : ./Model/Flowformer_sintel.pth
    eval_mode   : true | false
    cov_mode    : GT   | None
    device      : cuda | cpu | cuda:<idx>

```

```
type: TartanVODepth
args:
    weight      : ./Model/TartanVO_depth_cov.pth
    eval_mode   : true | false
    cov_mode    : Est  | None | GT
    device      : cpu  | cuda | cuda:<idx>
```

```
type: GTDepth
args:
```

## Match Estimator

```
type: FlowFormerMatcher
args:
    weight      : ./Model/MACVO_FrontendCov.pth
    eval_mode   : true | false
    cov_mode    : Est
    device      : cuda | cpu | cuda:<idx>
```

```
type: FlowFormerMatcher
args:
    weight      : ./Model/Flowformer_sintel.pth
    eval_mode   : true | false
    cov_mode    : GT   | None
    device      : cuda | cpu | cuda:<idx>
```

```
type: TartanVOMatcher
args:
    weight      : ./Model/205001_RAFTCov_b.pth
    eval_mode   : true | false
    cov_mode    : Est  | GT | None
    device      : cuda | cuda:<idx> # Only support cuda!
```

```
type: GTMatcher
args:
```

## Covariance Model

```
type: MatchCovariance
kwargs:
    device              : cpu | cuda | cuda:<idx>
    diag                : true | false  # false for highest performance
    match_cov_default   : 0.25
    kernel_size         : 31
    min_flow_cov        : 0.25
    min_depth_cov       : 0.05
```

```
type: GaussianMixtureCovariance
kwargs:
    match_cov_default   : 0.25
    kernel_size         : 31
    min_flow_cov        : 0.25
    min_depth_cov       : 0.05
```

```
type: DepthCovariance
kwargs:
```

```
type: NoCovariance
kwargs:
```

## Keypoint Selector

```
type: RandomSelector
args:
    mask_width: 32
```

```
type: GridSelector
args:
    mask_width: 32
    device    : cpu | cuda | cuda:<idx>
```

```
type: SparseGradienSelector
args:
    mask_width: 32
    grad_std  : 3.0
    nms_size  : 7
```

```
type: CovAwareSelector
args:
    device      : cpu | cuda | cuda:<idx>
    kernel_size : 7
    mask_width  : 32
    max_depth   : auto | <float>
    max_depth_cov: 100.0
    max_match_cov: 100.0
```

## Motion Model

```
type: TartanMotionNet
args:
    weight      : ./Model/MACVO_posenet.pkl
    eval_mode   : true | false
```

```
type: GTMotionwithNoise
args:
    noise_std   : 0.1
```

```
type: NaiveMotionModel
args:
```

```
type: GTMotionModel
args:
```

## Outlier Filter

```
type: FilterCompose
args:
    filter_args:
        - <OTHER_OUTLIER_FILTER1>
        - <OTHER_OUTLIER_FILTER2>
        - ...
```

```
type: SimpleDepthFilter
args:
    min_depth: 0.05
    max_depth: auto | <float>
```

```
type: CovarianceSanityFilter
args:
```

```
type: LikelyFrontOfCamFilter
args:
```

## Post Process (Map Elaboration)

```
type: Naive
args:
```

```
type: MotionInterpolate
args:
```

## Keyframe Selector

```
type: AllKeyframe
args:
```

```
type: UniformKeyframe
args:
    keyframe_freq   : 10
```

## Optimizer

```
type: TwoFramePoseOptimizer
args:
    device      : cpu | cuda | cuda:<idx>
    vectorize   : true | false 
    # False will be significantly slower! (& takes less RAM)
    parallel    : true | false
    # True will spawn a background process to run optimization loop. recommended to use `cpu` when
    # setting parallel=true to avoid resource contention on CUDA cores.
```

* Reprojection error with BA, this setup does not use depth information
```
type: ReprojectBAOptimizer
args:
    device: cpu
    vectorize: true
    parallel: true
    use_fancy_cov: false
    constraint: none
```