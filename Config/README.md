# Config Customization

## Config Syntax & Custom Tags

We used the `yaml` file format for config with some slight enhancement - the `!include`, `!none` and `!flatten` tags.

When using `Utility.Config.load_config` to read `yaml` file, the parser will have following action when the aforementioned tags are met:

* `!include <PATH>.yaml` will read the yaml file specified as the argument and directly concatenate the content into the position of `!include` tag.

* `!flatten` is used directly on a sequence (list) in yaml and will flatten the nested sequence into a "flat" sequence.


## Module and Interface

An **Interface** specifies a component with certain methods and properties. A **Module** is the actual implementation of an interface and follows the type specification provided by the interface. In MACVO, we strictly followed the interface and wrote *implementation-agnostic* code. Therefore, all implementations of interfaces are interchangable without any breaking change.

Currently the modules available / used by MAC-VO are

Interface           |   Description
---|---
`IKeypointSelector `  |   Select keypoint to track
`IMatchEstimator   `  |   Estimate dense optical flow and (optionally) the uncertainty of flow
`IDepthEstimator   `  |   Estimate dense depth and (optionally) the uncertainty of depth
`IMotionModel      `  |   Provide initial guess for the pose graph optimization
`IObservationFilter`  |   Filter out 'bad' / 'ill-defined' observations for tracking
`IObservationCov   `  |   Coverting 2D uncertainty to 3x3 spatial covariance matrix.
`IKeyframeSelector `  |   Selecting keyframe, poses at non-keyframe are interpolated
`IMapProcessor     `  |   Pose-process the map, perform smoothing / interpolation

The implementation of module is loaded **dynamically** following the config file, usually using the class method 

```python
Interface.instantiate("implementation_class_name", *args, **kwargs)
```

## Currently Available Modules & Config Spec

See [ConfigSpec.md](/Config/ConfigSpec.md).
